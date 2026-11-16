// Video decoder for mtmd in the browser.
//
// A browser decodes video itself, and only on the page (HTMLVideoElement), so
// packages/node-llama-wasm/index.js samples the frames there and stages them
// as a "frame pack": a header followed by the frames as RGB24. This decoder
// reads that pack; mtmd then picks the frames and the frame size the model
// expects from it, as it does with any other decoder.
//
// It is linked in as mtmd's built-in decoder (the slot of the ffmpeg one,
// which needs subprocesses), see src/wasm/CMakeLists.txt.
//
// Pack layout, little endian:
//   char[8] "LNVFRM01", u32 width, u32 height, f32 fps, u32 n_frames,
//   then n_frames * width * height * 3 bytes.

#include "mtmd-helper.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

const char   PACK_MAGIC[8]    = {'L', 'N', 'V', 'F', 'R', 'M', '0', '1'};
const size_t PACK_HEADER_SIZE = 24;

struct frame_pack {
    FILE *               file = nullptr; // a pack given by path is read frame by frame
    std::vector<uint8_t> mem;            // a pack given as a buffer
    size_t               mem_pos = 0;

    uint32_t width    = 0;
    uint32_t height   = 0;
    uint32_t n_frames = 0;
    uint32_t next     = 0;

    std::vector<uint8_t> frame; // a frame at the pack's size, before scaling

    ~frame_pack() {
        if (file) {
            fclose(file);
        }
    }

    size_t frame_size() const {
        return (size_t) width * height * 3;
    }

    bool read(uint8_t * dst, size_t n) {
        if (file) {
            return fread(dst, 1, n, file) == n;
        }
        if (mem.size() - mem_pos < n) {
            return false;
        }
        memcpy(dst, mem.data() + mem_pos, n);
        mem_pos += n;
        return true;
    }

    bool skip(size_t n) {
        if (file) {
            return fseek(file, (long) n, SEEK_CUR) == 0;
        }
        if (mem.size() - mem_pos < n) {
            return false;
        }
        mem_pos += n;
        return true;
    }
};

uint32_t read_u32(const uint8_t * p) {
    return (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24;
}

// Bilinear; the page already scaled the frames down to about the size models take
void scale_rgb(const uint8_t * src, uint32_t sw, uint32_t sh, uint8_t * dst, uint32_t dw, uint32_t dh) {
    for (uint32_t y = 0; y < dh; y++) {
        const float    fy = ((float) y + 0.5f) * sh / dh - 0.5f;
        const int32_t  y0 = fy < 0.0f ? 0 : (int32_t) fy;
        const int32_t  y1 = y0 + 1 < (int32_t) sh ? y0 + 1 : (int32_t) sh - 1;
        const float    wy = fy < 0.0f ? 0.0f : fy - (float) y0;
        for (uint32_t x = 0; x < dw; x++) {
            const float   fx = ((float) x + 0.5f) * sw / dw - 0.5f;
            const int32_t x0 = fx < 0.0f ? 0 : (int32_t) fx;
            const int32_t x1 = x0 + 1 < (int32_t) sw ? x0 + 1 : (int32_t) sw - 1;
            const float   wx = fx < 0.0f ? 0.0f : fx - (float) x0;
            const uint8_t * p00 = src + ((size_t) y0 * sw + x0) * 3;
            const uint8_t * p01 = src + ((size_t) y0 * sw + x1) * 3;
            const uint8_t * p10 = src + ((size_t) y1 * sw + x0) * 3;
            const uint8_t * p11 = src + ((size_t) y1 * sw + x1) * 3;
            uint8_t * out = dst + ((size_t) y * dw + x) * 3;
            for (int c = 0; c < 3; c++) {
                const float top = p00[c] + (p01[c] - p00[c]) * wx;
                const float bot = p10[c] + (p11[c] - p10[c]) * wx;
                out[c] = (uint8_t) (top + (bot - top) * wy + 0.5f);
            }
        }
    }
}

void * pack_open(void *, const char * path, const unsigned char * buf, size_t len,
                 mtmd_helper_video_stream_info * info) {
    auto * v = new frame_pack();
    uint8_t header[PACK_HEADER_SIZE];
    bool ok = false;
    if (path && (v->file = fopen(path, "rb"))) {
        ok = v->read(header, sizeof(header));
    } else if (buf && len >= sizeof(header)) {
        v->mem.assign(buf, buf + len); // buf is only valid during this call
        ok = v->read(header, sizeof(header));
    }
    if (!ok || memcmp(header, PACK_MAGIC, sizeof(PACK_MAGIC)) != 0) {
        // not a frame pack: an encoded video (or a webp image) that did not go through the page
        delete v;
        return nullptr;
    }

    float fps = 0.0f;
    memcpy(&fps, header + 16, sizeof(fps));
    v->width    = read_u32(header + 8);
    v->height   = read_u32(header + 12);
    v->n_frames = read_u32(header + 20);
    if (v->width == 0 || v->height == 0 || v->n_frames == 0 || !(fps > 0.0f)) {
        delete v;
        return nullptr;
    }

    info->width    = v->width;
    info->height   = v->height;
    info->fps      = fps;
    info->n_frames = (int32_t) v->n_frames;
    info->duration = (float) v->n_frames / fps;
    return v;
}

bool pack_read_frame(void * state, unsigned char * rgb, uint32_t width, uint32_t height) {
    auto * v = static_cast<frame_pack *>(state);
    if (v->next >= v->n_frames) {
        return false;
    }
    v->next++;
    if (!rgb) {
        return v->skip(v->frame_size());
    }
    if (width == v->width && height == v->height) {
        return v->read(rgb, v->frame_size());
    }
    v->frame.resize(v->frame_size());
    if (!v->read(v->frame.data(), v->frame.size())) {
        return false;
    }
    scale_rgb(v->frame.data(), v->width, v->height, rgb, width, height);
    return true;
}

void pack_close(void * state) {
    delete static_cast<frame_pack *>(state);
}

} // namespace

// mtmd's built-in decoder (declared in mtmd-helper-common.h, used with MTMD_VIDEO)
const mtmd_helper_video_decoder * mtmd_helper_video_decoder_ffmpeg() {
    static const mtmd_helper_video_decoder decoder = {
        /* open       */ pack_open,
        /* start      */ nullptr, // mtmd picks the frames by index
        /* read_frame */ pack_read_frame,
        /* read_audio */ nullptr, // frames only
        /* close      */ pack_close,
        /* user_data  */ nullptr,
    };
    return &decoder;
}

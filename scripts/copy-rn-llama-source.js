#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

// Colors for output
const colors = {
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  reset: "\x1b[0m",
};

function log(message, color = "reset") {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

log("Starting copy of llama.rn source files...", "green");

// Define source and destination directories
const SRC_DIR = "src/llama.rn/cpp";
const DEST_DIR = "src/rn-llama";

// Files to copy from llama.rn/cpp
const FILES_TO_COPY = [
  "anyascii.h",
  "anyascii.c",
  "rn-llama.h",
  "rn-llama.cpp",
  "rn-completion.h",
  "rn-completion.cpp",
  "rn-mtmd.hpp",
  "rn-tts.h",
  "rn-tts.cpp",
  // Parallel decoding support
  "rn-common.hpp",
  "rn-slot.h",
  "rn-slot.cpp",
  "rn-slot-manager.h",
  "rn-slot-manager.cpp",
];

try {
  // Create destination directory
  if (!fs.existsSync(DEST_DIR)) {
    fs.mkdirSync(DEST_DIR, { recursive: true });
  }

  // Clear destination directory
  if (fs.existsSync(DEST_DIR)) {
    fs.rmSync(DEST_DIR, { recursive: true, force: true });
    fs.mkdirSync(DEST_DIR, { recursive: true });
  }

  // Initialize and update llama.rn submodule if needed
  log("Ensuring llama.rn submodule is initialized...", "yellow");
  const headerPath = path.join(SRC_DIR, "rn-llama.h");

  if (!fs.existsSync(headerPath)) {
    log("Initializing llama.rn submodule...");
    execSync("git submodule init src/llama.rn", { stdio: "inherit" });
    execSync("git submodule update --recursive src/llama.rn", { stdio: "inherit" });
  }

  // Copy files and remove lm_ and LM_ prefixes
  FILES_TO_COPY.forEach((file) => {
    const srcPath = path.join(SRC_DIR, file);
    const destPath = path.join(DEST_DIR, file);

    if (fs.existsSync(srcPath)) {
      log(`Copying and processing ${file}...`, "yellow");

      // Read the file and process it to remove lm_ and LM_ prefixes
      let content = fs.readFileSync(srcPath, "utf8");
      content = content.replace(/lm_ggml/g, "ggml");
      content = content.replace(/LM_GGML/g, "GGML");
      content = content.replace(/lm_gguf/g, "gguf");
      content = content.replace(/LM_GGUF/g, "GGUF");
      content = content.replace(
        /mtmd_decode_use_non_causal\(mtmd_ctx\)/g,
        "mtmd_decode_use_non_causal(mtmd_ctx, nullptr)",
      );
      content = content.replace(
        /mtmd::bitmap bmp\(mtmd_helper_bitmap_init_from_buf\(mtmd_wrapper->mtmd_ctx, media_data\.data\(\), media_data\.size\(\)\)\);/g,
        [
          "auto bmp_res = mtmd_helper_bitmap_init_from_buf(mtmd_wrapper->mtmd_ctx, media_data.data(), media_data.size(), false);",
          "            mtmd::bitmap bmp(bmp_res.bitmap);",
          "            mtmd_helper::video_ptr video(bmp_res.video_ctx);",
          "            if (video) {",
          "                bitmaps.entries.clear();",
          '                throw std::runtime_error("Video media is not supported yet");',
          "            }",
        ].join("\n"),
      );
      content = content.replace(
        /mtmd::bitmap bmp\(mtmd_helper_bitmap_init_from_file\(mtmd_wrapper->mtmd_ctx, media_path\.c_str\(\)\)\);/g,
        [
          "auto bmp_res = mtmd_helper_bitmap_init_from_file(mtmd_wrapper->mtmd_ctx, media_path.c_str(), false);",
          "            mtmd::bitmap bmp(bmp_res.bitmap);",
          "            mtmd_helper::video_ptr video(bmp_res.video_ctx);",
          "            if (video) {",
          "                bitmaps.entries.clear();",
          '                throw std::runtime_error("Video media is not supported yet");',
          "            }",
        ].join("\n"),
      );

      // Write the processed content to destination
      fs.writeFileSync(destPath, content);

      log(`✓ ${file} processed and copied to ${destPath}`, "green");
    } else {
      log(`✗ Source file ${srcPath} not found!`, "red");
      process.exit(1);
    }
  });

  // Recursively copy the codec.cpp tree (vendored in llama.rn at cpp/codec)
  // with the same lm_ggml/LM_GGML prefix transforms.
  // tts_runner{,_flow}.cpp are codec.cpp's reference host loop; they pull in
  // codec_example_* helpers not shipped here. rn-tts drives its own AR loop
  // via codec_lm_*, so drop them (matches llama.rn's CMake exclusions).
  const CODEC_EXCLUDE = new Set(["tts_runner.cpp", "tts_runner_flow.cpp"]);
  const copyDirProcessed = (srcDir, destDir) => {
    fs.mkdirSync(destDir, { recursive: true });
    fs.readdirSync(srcDir, { withFileTypes: true }).forEach((entry) => {
      const srcPath = path.join(srcDir, entry.name);
      const destPath = path.join(destDir, entry.name);
      if (entry.isDirectory()) {
        copyDirProcessed(srcPath, destPath);
      } else if (CODEC_EXCLUDE.has(entry.name)) {
        log(`Skipping ${entry.name} (reference host loop, not used)`, "yellow");
      } else if (/\.(c|cc|cpp|h|hpp|inc)$/.test(entry.name)) {
        let content = fs.readFileSync(srcPath, "utf8");
        content = content.replace(/lm_ggml/g, "ggml");
        content = content.replace(/LM_GGML/g, "GGML");
        content = content.replace(/lm_gguf/g, "gguf");
        content = content.replace(/LM_GGUF/g, "GGUF");
        fs.writeFileSync(destPath, content);
      }
    });
  };

  const codecSrc = path.join(SRC_DIR, "codec");
  if (fs.existsSync(codecSrc)) {
    log("Copying and processing codec.cpp sources...", "yellow");
    copyDirProcessed(codecSrc, path.join(DEST_DIR, "codec"));
    log("✓ codec sources processed and copied to src/rn-llama/codec", "green");
  } else {
    log("✗ codec sources not found in llama.rn (cpp/codec)!", "red");
    process.exit(1);
  }

  log("All llama.rn source files copied and processed successfully!", "green");
  log("Note: lm_ggml and LM_GGML prefixes have been removed from all copied files.", "yellow");
} catch (error) {
  log(`Error: ${error.message}`, "red");
  process.exit(1);
}

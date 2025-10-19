
import type {
  ChatMessage,
} from './binding'

export const MTMD_DEFAULT_MEDIA_MARKER = '<__media__>'

export const formatMediaChat = (messages: ChatMessage[] | undefined): {
  messages: ChatMessage[] | undefined
  has_media: boolean
  media_paths?: string[]
} => {
  if (!messages)
    return {
      messages,
      has_media: false,
    }
  const mediaPaths: string[] = []
  return {
    messages: messages.map((msg) => {
      if (Array.isArray(msg.content)) {
        const content = msg.content.map((part) => {
          // Handle multimodal content
          if (part.type === 'image_url') {
            let path = part.image_url?.url || ''
            mediaPaths.push(path)
            return {
              type: 'text',
              text: MTMD_DEFAULT_MEDIA_MARKER,
            }
          } else if (part.type === 'input_audio') {
            const { input_audio: audio } = part
            if (!audio) throw new Error('input_audio is required')

            const { format } = audio
            if (format != 'wav' && format != 'mp3') {
              throw new Error(`Unsupported audio format: ${format}`)
            }
            if (audio.url) {
              const path = audio.url.replace(/file:\/\//, '')
              mediaPaths.push(path)
            } else if (audio.data) {
              mediaPaths.push(audio.data)
            }
            return {
              type: 'text',
              text: MTMD_DEFAULT_MEDIA_MARKER,
            }
          }
          return part
        })

        return {
          ...msg,
          content,
        }
      }
      return msg
    }),
    has_media: mediaPaths.length > 0,
    media_paths: mediaPaths,
  }
}
const fs = require('fs')
const http = require('http')
const path = require('path')

const rootDir = path.resolve(__dirname, '..')
const port = Number(process.env.PORT || 8088)

const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url || '/', `http://${req.headers.host}`)
  const pathname =
    url.pathname === '/'
      ? '/test/web/llama-node-wasm.html'
      : decodeURIComponent(url.pathname)
  const filePath = path.resolve(rootDir, `.${pathname}`)

  if (!filePath.startsWith(rootDir)) {
    res.writeHead(403)
    res.end('Forbidden')
    return
  }

  fs.readFile(filePath, (error, data) => {
    if (error) {
      res.writeHead(404)
      res.end('Not found')
      return
    }

    res.writeHead(200, {
      'Content-Type':
        contentTypes[path.extname(filePath)] || 'application/octet-stream',
      'Cache-Control': 'no-store',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    })
    res.end(data)
  })
})

server.listen(port, () => {
  console.log(`WASM test server: http://localhost:${port}/`)
})

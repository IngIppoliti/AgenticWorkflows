"""
Live dashboard for PlanState — zero dependencies, uses http.server.

Start BEFORE the workflow:
    python workflow_agents/dashboard.py

Open http://127.0.0.1:9090 in a browser.
The dashboard polls plan_state.json every 1.5s and renders the plan.
"""

import os
import socketserver
from http import server

HOST = "127.0.0.1"
PORT = 9090
STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "plan_state.json"
)

PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Workflow Dashboard</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #f4efdc;
    font-family: 'Courier New', monospace;
    padding: 2rem;
    color: #222;
  }
  h1 {
    font-size: 2rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    border: 4px solid #222;
    display: inline-block;
    padding: 0.5rem 1.5rem;
    background: #fff;
    box-shadow: 6px 6px 0 #222;
    margin-bottom: 2rem;
  }
  .plan-prompt {
    font-size: 1.1rem;
    background: #fff;
    border: 3px solid #222;
    padding: 0.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 4px 4px 0 #222;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    background: #fff;
    border: 4px solid #222;
    box-shadow: 8px 8px 0 #222;
  }
  th {
    background: #222;
    color: #fff;
    text-align: left;
    padding: 0.6rem 0.8rem;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  td {
    padding: 0.6rem 0.8rem;
    border-bottom: 2px solid #ddd;
    vertical-align: top;
  }
  .status-badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: bold;
    padding: 0.2rem 0.6rem;
    border: 2px solid #222;
    text-transform: uppercase;
  }
  .status-pending   { background: #eee; }
  .status-progress  { background: #fadb5f; }
  .status-completed { background: #7acb7a; }
  .status-failed    { background: #e06c6c; color: #fff; }
  .worker { font-weight: bold; }
  .output-cell {
    max-width: 400px;
    word-break: break-word;
    font-size: 0.85rem;
    cursor: pointer;
  }
  .output-cell .truncated { display: inline; }
  .output-cell .full { display: none; }
  .output-cell.expanded .truncated { display: none; }
  .output-cell.expanded .full { display: inline; }
  .progress-bar-wrap {
    margin-top: 1.5rem;
    border: 3px solid #222;
    height: 28px;
    background: #ddd;
    box-shadow: 4px 4px 0 #222;
  }
  .progress-bar-fill {
    height: 100%;
    background: #7acb7a;
    transition: width 0.5s;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 0.8rem;
  }
  .footer {
    margin-top: 2rem;
    font-size: 0.8rem;
    color: #666;
  }
  .empty { padding: 2rem; text-align: center; color: #888; }
</style>
</head>
<body>
<h1>⚡ Workflow</h1>
<div id="prompt" class="plan-prompt">Waiting for plan...</div>
<table>
  <thead>
    <tr><th>#</th><th>Status</th><th>Task</th><th>Worker</th><th>Output</th></tr>
  </thead>
  <tbody id="steps"></tbody>
</table>
<div class="progress-bar-wrap"><div id="progress" class="progress-bar-fill" style="width:0%">0%</div></div>
<div id="final-section" style="display:none; margin-top: 2rem;">
  <h2 style="font-size:1.3rem; text-transform:uppercase; letter-spacing:2px; border:4px solid #222; display:inline-block; padding:0.4rem 1.2rem; background:#fff; box-shadow:5px 5px 0 #222; margin-bottom:1rem;">
    📋 Final Result
  </h2>
  <pre id="final-output" style="
    background:#fff; border:4px solid #222; box-shadow:8px 8px 0 #222;
    padding:1.2rem; font-family:'Courier New',monospace; font-size:0.85rem;
    line-height:1.5; white-space:pre-wrap; word-break:break-word;
    max-height:none; overflow:visible;
  "></pre>
</div>
<div id="footer" class="footer">Polling plan_state.json...</div>

<script>
async function refresh() {
  try {
    const r = await fetch('/state');
    if (!r.ok) throw new Error(r.status);
    const data = await r.json();
    render(data);
  } catch { /* file not ready yet */ }
}

function render(data) {
  const entries = data.entries || [];
  const total = entries.length || 1;
  const done = entries.filter(e => e.status === 'completed').length;
  const failed = entries.filter(e => e.status === 'failed').length;
  const pct = Math.round((done + failed) / total * 100);

  document.getElementById('prompt').textContent = data.prompt || '(no plan)';

  const tbody = document.getElementById('steps');
  tbody.innerHTML = entries.length === 0
    ? '<tr><td colspan="5" class="empty">No steps yet</td></tr>'
    : entries.map((e, i) => {
        const cls = e.status === 'in_progress' ? 'progress'
                : e.status === 'completed' ? 'completed'
                : e.status === 'failed' ? 'failed'
                : 'pending';
        const icon = e.status === 'in_progress' ? '🔄'
                : e.status === 'completed' ? '✅'
                : e.status === 'failed' ? '❌'
                : '⏳';
        const hasFullOutput = e.final_output && e.final_output.length > 120;
        const truncated = hasFullOutput
          ? escapeHtml(e.final_output.slice(0, 120)) + '…'
          : escapeHtml(e.final_output || e.error_message || '—');
        const fullText = hasFullOutput
          ? escapeHtml(e.final_output)
          : '';
        const fullHtml = hasFullOutput
          ? '<span class="truncated">' + truncated + '</span><span class="full">' + fullText + '</span>'
          : truncated;
        return `<tr>
          <td>${e.step_number}</td>
          <td><span class="status-badge status-${cls}">${icon} ${e.status}</span></td>
          <td>${escapeHtml(e.task_description)}</td>
          <td class="worker">${e.assigned_worker || '—'}</td>
          <td class="output-cell" onclick="this.classList.toggle('expanded')">${fullHtml}</td>
        </tr>`;
      }).join('');

  const bar = document.getElementById('progress');
  bar.style.width = pct + '%';
  bar.textContent = done + '/' + total + ' done' + (failed ? ' (' + failed + ' failed)' : '');

  // ── Final output section (full text, no truncation) ──────────────
  const finalSection = document.getElementById('final-section');
  const finalPre = document.getElementById('final-output');
  if (data.final_output) {
    finalSection.style.display = 'block';
    finalPre.textContent = data.final_output;
  } else {
    finalSection.style.display = 'none';
  }
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

setInterval(refresh, 1500);
refresh();
</script>
</body>
</html>
"""


class Handler(server.BaseHTTPRequestHandler):
    """Serve the HTML page and the JSON state."""

    def do_GET(self):
        if self.path == "/state":
            self._serve_json()
        else:
            self._serve_html()

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE.encode("utf-8"))

    def _serve_json(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = "{}"

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(content.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def log_message(self, fmt, *args):
        pass  # quieter output


def main():
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer((HOST, PORT), Handler) as httpd:
            print(f"🌐 Dashboard at http://{HOST}:{PORT}")
            print("   Press Ctrl+C to stop.")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n   Dashboard stopped.")
    except OSError as exc:
        if "10048" in str(exc):
            print(
                f"❌ Port {PORT} is already in use.\n"
                f"   Another dashboard may be running.\n"
                f"   Stop it with: taskkill /F /IM python.exe\n"
                f"   Or kill the process using port {PORT}:"
                f" netstat -ano | findstr :{PORT}"
            )
        else:
            raise


if __name__ == "__main__":
    main()

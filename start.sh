#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# AI Tutor — start all services
#
# Usage:
#   ./start.sh            start everything
#   ./start.sh stop       stop all running services
#   ./start.sh restart    stop then start
#   ./start.sh status     show what is running
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_ROOT/.venv"
PYTHON="$VENV/bin/python"
LOG_DIR="$REPO_ROOT/logs"
PID_DIR="$REPO_ROOT/.pids"

# Vertex endpoint config lives in a single .env file that is sourced here.
# Rotate the endpoint by editing configs/vertex.env — no code changes needed.
VERTEX_ENV_FILE="$REPO_ROOT/configs/vertex.env"
VERTEX_ENV_EXAMPLE="$REPO_ROOT/configs/vertex.env.example"

HTML_PORT="${HTML_PORT:-8080}"
GRADIO_PORT="${GRADIO_PORT:-7860}"
API_PORT="${API_PORT:-8000}"

HTML_SRC="$REPO_ROOT/modules/ui_module_html/src/server.py"
GRADIO_SRC="$REPO_ROOT/modules/ui_module/src/learn_tab_app.py"
API_SRC="$REPO_ROOT/modules/ui_module/src/learn_api.py"

# colours
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

load_vertex_env() {
    if [[ -f "$VERTEX_ENV_FILE" ]]; then
        set -a
        # shellcheck source=/dev/null
        source "$VERTEX_ENV_FILE"
        set +a
    else
        warn "configs/vertex.env not found — copy $VERTEX_ENV_EXAMPLE and fill it in."
        warn "Services may fail on the first Vertex call until this is done."
    fi
}

# ── Helpers ───────────────────────────────────────────────────────────────────
log()    { echo -e "${CYAN}[aitutor]${NC} $*"; }
ok()     { echo -e "${GREEN}[  OK  ]${NC} $*"; }
warn()   { echo -e "${YELLOW}[ WARN ]${NC} $*"; }
err()    { echo -e "${RED}[ ERR  ]${NC} $*" >&2; }

mkdir -p "$LOG_DIR" "$PID_DIR"

pid_file() { echo "$PID_DIR/$1.pid"; }

is_running() {
    local pf; pf=$(pid_file "$1")
    [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null
}

stop_service() {
    local name="$1"
    local pf; pf=$(pid_file "$name")
    if is_running "$name"; then
        kill "$(cat "$pf")" 2>/dev/null && rm -f "$pf"
        ok "Stopped $name"
    else
        warn "$name was not running"
        rm -f "$pf"
    fi
}

start_service() {
    local name="$1"
    local cmd=("${@:2}")
    local logfile="$LOG_DIR/${name}.log"
    local pf; pf=$(pid_file "$name")

    if is_running "$name"; then
        warn "$name already running (PID $(cat "$pf"))"
        return
    fi

    log "Starting $name …"
    nohup "${cmd[@]}" >> "$logfile" 2>&1 &
    echo $! > "$pf"
    sleep 1
    if is_running "$name"; then
        ok "$name started  (PID $(cat "$pf"))  →  log: $logfile"
    else
        err "$name failed to start — check $logfile"
        cat "$logfile" | tail -20
        exit 1
    fi
}

wait_for_port() {
    local name="$1" port="$2"
    local tries=0
    printf "${CYAN}[aitutor]${NC} Waiting for %s on port %s " "$name" "$port"
    until nc -z 127.0.0.1 "$port" 2>/dev/null; do
        printf "."
        sleep 1
        tries=$((tries+1))
        if [[ $tries -ge 60 ]]; then
            echo ""
            err "$name did not become ready on port $port in 60s"
            return 1
        fi
    done
    echo ""
    ok "$name is ready on http://127.0.0.1:$port"
}

# ── Commands ──────────────────────────────────────────────────────────────────
cmd_stop() {
    log "Stopping all AI Tutor services …"
    stop_service "html_frontend"
    stop_service "gradio_ui"
    stop_service "learn_api"
    ok "All services stopped."
}

cmd_status() {
    echo ""
    echo -e "  Service           PID        Port   Status"
    echo -e "  ─────────────────────────────────────────────"
    for svc in html_frontend gradio_ui learn_api; do
        local pf; pf=$(pid_file "$svc")
        if is_running "$svc"; then
            local pid; pid=$(cat "$pf")
            printf "  %-18s %-10s %-7s ${GREEN}running${NC}\n" "$svc" "$pid" ""
        else
            printf "  %-18s %-10s %-7s ${RED}stopped${NC}\n" "$svc" "-" ""
        fi
    done
    echo ""
}

cmd_start() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║         AI Tutor — Starting Services         ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
    echo ""

    # Sanity checks
    if [[ ! -f "$PYTHON" ]]; then
        err "Virtual environment not found at $VENV"
        err "Run: python3 -m venv $VENV && source $VENV/bin/activate && pip install -r requirements.txt"
        exit 1
    fi

    load_vertex_env
    if [[ -n "${VERTEX_ENDPOINT_ID:-}" ]]; then
        ok "Vertex endpoint: ${VERTEX_ENDPOINT_ID} (${VERTEX_LOCATION:-?})"
        if [[ -n "${VERTEX_API_ENDPOINT:-}" ]]; then
            ok "PSC host:        ${VERTEX_API_ENDPOINT}"
        fi
    fi

    # 1. FastAPI Learn API (port 8000) — Vertex env flows in via the parent shell.
    start_service "learn_api" \
        "$PYTHON" -m uvicorn learn_api:app \
        --app-dir "$REPO_ROOT/modules/ui_module/src" \
        --host 127.0.0.1 --port "$API_PORT"

    # 2. Gradio Tutor UI (port 7860)
    start_service "gradio_ui" \
        "$PYTHON" "$GRADIO_SRC"

    # 3. HTML Frontend (port 8080) — pass Gradio URL as env var
    GRADIO_URL="http://127.0.0.1:$GRADIO_PORT" \
    start_service "html_frontend" \
        "$PYTHON" -m uvicorn server:app \
        --app-dir "$REPO_ROOT/modules/ui_module_html/src" \
        --host 127.0.0.1 --port "$HTML_PORT"

    echo ""
    # Wait for each port to be ready
    wait_for_port "learn_api"     "$API_PORT"
    wait_for_port "gradio_ui"     "$GRADIO_PORT"
    wait_for_port "html_frontend" "$HTML_PORT"

    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║            All services are UP!              ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  ${CYAN}🌐 Open in browser:${NC}  http://127.0.0.1:${HTML_PORT}"
    echo -e "  ${CYAN}🎓 Gradio tutor:${NC}     http://127.0.0.1:${GRADIO_PORT}"
    echo -e "  ${CYAN}⚙️  Learn API:${NC}        http://127.0.0.1:${API_PORT}"
    echo ""
    echo -e "  Logs are in: ${YELLOW}$LOG_DIR/${NC}"
    echo -e "  To stop:     ${YELLOW}./start.sh stop${NC}"
    echo ""
}

cmd_doctor() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   Vertex endpoint connectivity doctor        ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
    echo ""

    load_vertex_env

    local fail=0
    local project="${VERTEX_PROJECT_ID:-}"
    local region="${VERTEX_LOCATION:-}"
    local endpoint_id="${VERTEX_ENDPOINT_ID:-}"
    local api_host="${VERTEX_API_ENDPOINT:-}"

    if [[ -z "$project" || -z "$region" || -z "$endpoint_id" ]]; then
        err "VERTEX_PROJECT_ID / VERTEX_LOCATION / VERTEX_ENDPOINT_ID missing in $VERTEX_ENV_FILE"
        exit 1
    fi
    ok "Project:      $project"
    ok "Location:     $region"
    ok "Endpoint ID:  $endpoint_id"
    [[ -n "$api_host" ]] && ok "PSC host:     $api_host" || warn "VERTEX_API_ENDPOINT empty — will use public ${region}-aiplatform.googleapis.com"

    # 1) DNS resolution
    if [[ -n "$api_host" ]]; then
        echo ""; log "[1/4] Resolving DNS for $api_host …"
        if getent hosts "$api_host" >/dev/null 2>&1; then
            ok "DNS resolves: $(getent hosts "$api_host" | awk '{print $1}' | head -1)"
        else
            err "DNS does NOT resolve $api_host from this machine."
            err "→ Likely cause: endpoint is PRIVATE (PSC). You need to run from a VM/Workstation"
            err "  attached to the VPC that has the PSC forwarding rule for this endpoint."
            fail=1
        fi
    fi

    # 2) TCP reachability on 443
    if [[ -n "$api_host" ]] && [[ $fail -eq 0 ]]; then
        echo ""; log "[2/4] TCP 443 reachability to $api_host …"
        if timeout 5 bash -c ">/dev/tcp/$api_host/443" 2>/dev/null; then
            ok "TCP 443 is reachable."
        else
            err "TCP 443 timed out — DNS resolved but no route. VPC/firewall/PSC missing."
            fail=1
        fi
    fi

    # 3) ADC identity
    echo ""; log "[3/4] Google Application Default Credentials …"
    local adc_json
    adc_json=$("$PYTHON" - <<'PY' 2>&1 || true
try:
    import google.auth
    creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    from google.auth.transport.requests import Request
    creds.refresh(Request())
    acct = getattr(creds, "service_account_email", None) or getattr(creds, "_service_account_email", None) or "(user credentials)"
    print(f"ADC_OK project={project} identity={acct} has_token={bool(creds.token)}")
except Exception as e:
    print(f"ADC_FAIL {type(e).__name__}: {e}")
PY
    )
    if [[ "$adc_json" == ADC_OK* ]]; then
        ok "$adc_json"
    else
        err "$adc_json"
        err "→ Fix: gcloud auth application-default login  (or set GOOGLE_APPLICATION_CREDENTIALS)"
        fail=1
    fi

    # 4) Auth'd :predict round-trip (real test — same code path as the API)
    if [[ $fail -eq 0 ]]; then
        echo ""; log "[4/4] End-to-end :predict sanity call …"
        "$PYTHON" - <<PY
import os, sys, json
sys.path.insert(0, os.path.join("${REPO_ROOT}", "modules", "teacher_module", "src"))
from teacher_pdf_pipeline import load_vertex_endpoint_client
try:
    client = load_vertex_endpoint_client(
        project_id="${project}",
        location="${region}",
        endpoint_id="${endpoint_id}",
        api_endpoint="${api_host}" or None,
    )
    text = client.generate_text("Reply with the single word: OK", max_new_tokens=8)
    print(f"OK  generate_text() returned: {text!r}")
except Exception as e:
    print(f"ERR {type(e).__name__}: {e}")
    sys.exit(2)
PY
        if [[ $? -ne 0 ]]; then
            err "Prediction call failed — see error above."
            fail=1
        fi
    fi

    echo ""
    if [[ $fail -eq 0 ]]; then
        ok "All checks passed — services can reach this endpoint."
    else
        err "One or more checks failed. See messages above."
        exit 2
    fi
}

# ── Entrypoint ────────────────────────────────────────────────────────────────
case "${1:-start}" in
    start)   cmd_start   ;;
    stop)    cmd_stop    ;;
    restart) cmd_stop; sleep 1; cmd_start ;;
    status)  cmd_status  ;;
    doctor|check) cmd_doctor ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|doctor}"
        exit 1
        ;;
esac

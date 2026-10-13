"""
Ollama Server Control — manages the Ollama instance used by T1.

Supports both direct API access (when Ollama is reachable) and SSH-based
control (when Ollama runs on a remote server like localhost).

Commands:
    python3 -m token_optimiser.dashboard.ollama_control status
    python3 -m token_optimiser.dashboard.ollama_control start
    python3 -m token_optimiser.dashboard.ollama_control stop
    python3 -m token_optimiser.dashboard.ollama_control restart
    python3 -m token_optimiser.dashboard.ollama_control pull qwen2.5:3b
    python3 -m token_optimiser.dashboard.ollama_control list
    python3 -m token_optimiser.dashboard.ollama_control health
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ollama-control")


# ── Configuration ───────────────────────────────────────────────────────────────

@dataclass
class OllamaConfig:
    """Configuration for Ollama server control."""

    host: str = "localhost"
    """Ollama server host."""

    # DEFECT (CHG-2026-035): this was 8081 — the llama.cpp / Vulkan T1 port. So
    # every "Ollama health check" actually queried llama.cpp with Ollama's
    # `/api/tags` path, got HTTP 404, and reported the local model as DEAD. That
    # false negative is why T1 looked broken for so long: the health check was
    # measuring the wrong server on the wrong port, then falling through to an
    # SSH probe that also failed. Verified: curl to :11434/api/tags returns 200
    # and Ollama is live with llama3.1-16k; curl to :8081/api/tags returns 404
    # because that is llama.cpp, which speaks /v1/* not /api/*.
    port: int = 11434
    """Ollama API port. Ollama's default is 11434; 8081 is llama.cpp (T1)."""

    ssh_user: str = "warpuppy"
    """SSH user for remote control (if SSH is needed)."""

    ssh_host: str = "localhost"
    """SSH target host. Set to same as host for local control."""

    ssh_key: Optional[str] = None
    """SSH private key path. Uses default agent if None."""

    models_dir: str = "~/.ollama/models"
    """Ollama models directory on the server."""

    default_model: str = "qwen2.5:3b"
    """Default model to ensure is pulled."""

    use_ssh: bool = True
    """Whether to use SSH for control commands. If False, uses direct API."""

    ollama_binary: str = "ollama"
    """Ollama binary name/path on the server."""

    pull_timeout: int = 600
    """Max seconds to wait for a model pull."""

    health_check_timeout: int = 10
    """Timeout for health check requests."""

    # Dashboard-editable: these can be changed from the dashboard
    control_method: str = "ssh"  # "ssh" or "api" or "auto"
    """How to control Ollama: ssh (remote commands), api (direct HTTP), auto (try api first, fall back to ssh)"""


# Global config — dashboard will update this
ollama_config = OllamaConfig()


def set_control_method(method: str):
    ollama_config.control_method = method


def get_control_method() -> str:
    return ollama_config.control_method


# ── SSH helpers ────────────────────────────────────────────────────────────────

def _ssh_run(command: str, timeout: int = 60) -> Dict[str, Any]:
    """
    Run a command on the Ollama server via SSH.
    Returns { returncode, stdout, stderr, success }.
    """
    ssh_parts = ["ssh"]
    if ollama_config.ssh_key:
        ssh_parts.extend(["-i", ollama_config.ssh_key])
    ssh_parts.extend(["-o", "ConnectTimeout=10", "-o", "BatchMode=yes"])
    ssh_parts.append(f"{ollama_config.ssh_user}@{ollama_config.ssh_host}")
    ssh_parts.append(command)

    try:
        result = subprocess.run(
            ssh_parts,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "success": result.returncode == 0,
            "command": command,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "Command timed out",
            "success": False,
            "command": command,
        }
    except FileNotFoundError:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "SSH not available",
            "success": False,
            "command": command,
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e)[:200],
            "success": False,
            "command": command,
        }


def _ssh_run_sudo(command: str, timeout: int = 60) -> Dict[str, Any]:
    """Run a command with sudo on the Ollama server."""
    return _ssh_run(f"sudo {command}", timeout)


# ── Direct API helpers ──────────────────────────────────────────────────────────

def _api_request(method: str, path: str, data: Optional[Dict] = None,
                 timeout: int = 10) -> Dict[str, Any]:
    """
    Make a direct HTTP request to the Ollama API.
    """
    import urllib.request
    import urllib.error
    # DEFECT (CHG-2026-035): `import json as json_mod` lived inside the
    # `if data is not None:` block below, but json_mod is also used to parse
    # the RESPONSE. Every GET — including the health check's /api/tags — raised
    # UnboundLocalError, which the broad `except Exception` converted into
    # "unreachable". Ollama was therefore reported dead while serving 200s, and
    # the fallback SSH probe then also failed. A GET never needs this import to
    # be conditional.
    import json as json_mod

    url = f"http://{ollama_config.host}:{ollama_config.port}{path}"
    body = None
    headers = {}

    if data is not None:
        body = json_mod.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    try:
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response_body = resp.read().decode("utf-8")
            try:
                return {
                    "success": True,
                    "status_code": resp.status,
                    "data": json_mod.loads(response_body) if response_body else None,
                }
            except json_mod.JSONDecodeError:
                return {
                    "success": True,
                    "status_code": resp.status,
                    "data": response_body,
                }
    except urllib.error.HTTPError as e:
        return {
            "success": False,
            "status_code": e.code,
            "error": f"HTTP {e.code}: {e.reason}",
            "data": e.read().decode("utf-8", errors="replace") if e.fp else None,
        }
    except urllib.error.URLError as e:
        return {
            "success": False,
            "status_code": 0,
            "error": f"Connection failed: {e.reason}",
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "error": str(e)[:200],
        }


# ── Control commands ────────────────────────────────────────────────────────────

def get_status() -> Dict[str, Any]:
    """
    Get detailed Ollama server status.
    Returns server info, running state, model list, memory usage.
    """
    result = {
        "host": f"{ollama_config.host}:{ollama_config.port}",
        "control_method": ollama_config.control_method,
        "server_running": False,
        "api_reachable": False,
        "models": [],
        "memory_info": {},
        "error": None,
    }

    # Try direct API first
    if ollama_config.control_method in ("api", "auto"):
        api_result = _api_request("GET", "/api/tags")
        if api_result.get("success"):
            result["api_reachable"] = True
            result["server_running"] = True
            models_data = api_result.get("data", [])
            result["models"] = [
                {
                    "name": m.get("name", ""),
                    "digest": m.get("digest", "")[:12] + "..." if m.get("digest") else "",
                    "size": m.get("size", 0),
                    "modified_at": m.get("modified_at", "")[:10] if m.get("modified_at") else "",
                    "details": m.get("details", {}),
                }
                for m in (models_data if isinstance(models_data, list) else [])
            ]
            # Try to get system info
            info_result = _api_request("GET", "/api/version")
            if info_result.get("success"):
                result["version"] = info_result.get("data", {}).get("version", "unknown")
            else:
                # Try older endpoint
                ver_result = _api_request("GET", "/api/info")
                if ver_result.get("success"):
                    result["version"] = "unknown (older API)"
            # Memory estimation from model sizes
            total_size = sum(m.get("size", 0) for m in (api_result.get("data", []) or []))
            result["memory_info"] = {
                "total_model_size_bytes": total_size,
                "total_model_size_gb": round(total_size / (1024**3), 2),
                "model_count": len(result["models"]),
            }
            return result

    # Try SSH if API failed or control method is ssh
    if ollama_config.control_method in ("ssh", "auto"):
        ssh_status = _ssh_run("ollama list 2>/dev/null || echo 'OLLAMA_NOT_RUNNING'")
        if ssh_status["success"] and "OLLAMA_NOT_RUNNING" not in ssh_status["stdout"]:
            result["server_running"] = True
            result["control_method"] = "ssh"

            # Parse ollama list output
            lines = ssh_status["stdout"].split("\n")
            for line in lines[1:]:  # Skip header
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    size_str = parts[1] if len(parts) > 1 else "0B"
                    result["models"].append({
                        "name": name,
                        "size": size_str,
                        "digest": "",
                        "modified_at": "",
                        "details": {},
                    })

            # Get version via SSH
            ver = _ssh_run("ollama --version 2>/dev/null || echo 'unknown'")
            result["version"] = ver["stdout"].strip() or "unknown"

            # Estimate memory from model sizes
            total_size_bytes = 0
            for m in result["models"]:
                size_str = m.get("size", "0B")
                total_size_bytes += _parse_size(size_str)
            result["memory_info"] = {
                "total_model_size_bytes": total_size_bytes,
                "total_model_size_gb": round(total_size_bytes / (1024**3), 2),
                "model_count": len(result["models"]),
            }
        else:
            result["control_method"] = "ssh"
            result["error"] = ssh_status.get("stderr", "SSH command failed")[:200]

    # If still no info, try one more API attempt
    if not result["server_running"] and ollama_config.control_method == "auto":
        api_result2 = _api_request("GET", "/api/tags")
        if api_result2.get("success"):
            result["api_reachable"] = True
            result["server_running"] = True
            result["control_method"] = "api"
            models_data = api_result2.get("data", [])
            result["models"] = [
                {
                    "name": m.get("name", ""),
                    "digest": m.get("digest", "")[:12] + "..." if m.get("digest") else "",
                    "size": m.get("size", 0),
                }
                for m in (models_data if isinstance(models_data, list) else [])
            ]
            total_size = sum(m.get("size", 0) for m in (models_data or []))
            result["memory_info"] = {
                "total_model_size_gb": round(total_size / (1024**3), 2),
                "model_count": len(result["models"]),
            }

    return result


def _parse_size(size_str: str) -> int:
    """Parse a size string like '4.2 GB' or '3.5GB' or '500 MB' into bytes."""
    import re
    match = re.match(r"([\d.]+)\s*(GB|MB|KB|B)", size_str, re.IGNORECASE)
    if not match:
        return 0
    value = float(match.group(1))
    unit = match.group(2).upper()
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    return int(value * multipliers.get(unit, 1))


def start_server() -> Dict[str, Any]:
    """
    Start the Ollama server.
    On remote: uses SSH to start the ollama service.
    On local: uses direct API or systemctl.
    """
    result = {
        "action": "start",
        "method": ollama_config.control_method,
        "success": False,
        "message": "",
        "server_running_after": None,
    }

    if ollama_config.control_method in ("ssh", "auto"):
        # Try starting via SSH — first as user, then with sudo if needed
        ssh_out = _ssh_run("pgrep -f ollama serve || ollama serve &")
        if ssh_out["success"] or "OLLAMA" in ssh_out.get("stdout", ""):
            result["message"] = "Ollama serve command issued via SSH"
            result["success"] = True
        else:
            # Try with sudo
            ssh_sudo = _ssh_run_sudo("systemctl start ollama 2>/dev/null || ollama serve &")
            if ssh_sudo["success"]:
                result["message"] = "Ollama started via systemctl or direct command"
                result["success"] = True
            else:
                result["message"] = f"SSH start failed: {ssh_sudo.get('stderr', '')[:100]}"
    else:
        # Direct API — can't start server via API, check if it's already running
        api_check = _api_request("GET", "/api/tags")
        if api_check.get("success"):
            result["message"] = "Ollama is already running (API reachable)"
            result["success"] = True
        else:
            result["message"] = "Cannot start Ollama via API — server must be started externally"

    # Verify after start attempt
    time.sleep(3)
    status = get_status()
    result["server_running_after"] = status.get("server_running", False)
    result["models_after"] = len(status.get("models", []))
    return result


def stop_server() -> Dict[str, Any]:
    """Stop the Ollama server."""
    result = {
        "action": "stop",
        "method": ollama_config.control_method,
        "success": False,
        "message": "",
        "server_running_after": None,
    }

    if ollama_config.control_method in ("ssh", "auto"):
        # Kill ollama processes
        ssh_kill = _ssh_run_sudo("pkill -f 'ollama serve' 2>/dev/null; pkill -f ollama 2>/dev/null; echo DONE")
        if ssh_kill["success"]:
            result["message"] = "Ollama processes killed via SSH"
            result["success"] = True
        else:
            # Try systemctl
            ssh_sys = _ssh_run_sudo("systemctl stop ollama 2>/dev/null; echo DONE")
            if ssh_sys["success"]:
                result["message"] = "Ollama stopped via systemctl"
                result["success"] = True
            else:
                result["message"] = f"Stop failed: {ssh_kill.get('stderr', '')[:100]}"
    else:
        result["message"] = "Cannot stop Ollama via API — server must be stopped externally"

    time.sleep(2)
    status = get_status()
    result["server_running_after"] = status.get("server_running", False)
    return result


def restart_server() -> Dict[str, Any]:
    """Restart the Ollama server."""
    stop_result = stop_server()
    time.sleep(2)
    start_result = start_server()
    return {
        "action": "restart",
        "stop_result": stop_result,
        "start_result": start_result,
        "success": start_result.get("server_running_after", False),
    }


def pull_model(model_name: str, timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Pull a model from the Ollama registry.
    Returns progress updates and final status.
    """
    timeout = timeout or ollama_config.pull_timeout
    result = {
        "action": "pull",
        "model": model_name,
        "method": ollama_config.control_method,
        "success": False,
        "message": "",
        "progress": [],
        "final_status": None,
    }

    if ollama_config.control_method in ("ssh", "auto"):
        # SSH pull — capture output
        ssh_pull = _ssh_run(f"ollama pull {model_name}", timeout=timeout)
        if ssh_pull["success"]:
            result["success"] = True
            result["message"] = f"Model {model_name} pulled successfully via SSH"
            result["progress"] = [ssh_pull["stdout"]] if ssh_pull["stdout"] else []
        else:
            # Check if already pulled
            status = get_status()
            existing = [m["name"] for m in status.get("models", [])]
            if model_name in existing:
                result["success"] = True
                result["message"] = f"Model {model_name} already exists"
                result["progress"] = ["Model already present — no pull needed"]
            else:
                result["message"] = f"Pull failed: {ssh_pull.get('stderr', '')[:200]}"
                result["progress"] = [ssh_pull.get("stdout", ""), ssh_pull.get("stderr", "")][:20]
    else:
        # Direct API pull — streaming
        import urllib.request
        url = f"http://{ollama_config.host}:{ollama_config.port}/api/pull"
        import json as json_mod
        req_data = json_mod.dumps({"name": model_name}).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=req_data, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                chunks = []
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    chunks.append(chunk.decode("utf-8", errors="replace"))
                result["success"] = True
                result["message"] = f"Model {model_name} pulled successfully via API"
                result["progress"] = ["".join(chunks)[:500]]
        except Exception as e:
            # Check if already pulled
            status = get_status()
            existing = [m["name"] for m in status.get("models", [])]
            if model_name in existing:
                result["success"] = True
                result["message"] = f"Model {model_name} already exists"
            else:
                result["message"] = f"Pull failed: {str(e)[:200]}"

    # Final status
    time.sleep(2)
    final_status = get_status()
    result["final_status"] = {
        "server_running": final_status.get("server_running"),
        "model_count": len(final_status.get("models", [])),
        "has_model": any(m["name"] == model_name for m in final_status.get("models", [])),
    }
    return result


def list_models() -> Dict[str, Any]:
    """List all models on the Ollama server."""
    status = get_status()
    return {
        "models": status.get("models", []),
        "count": len(status.get("models", [])),
        "total_size_gb": status.get("memory_info", {}).get("total_model_size_gb", 0),
        "server_running": status.get("server_running"),
        "control_method": status.get("control_method"),
    }


def check_health() -> Dict[str, Any]:
    """
    Quick health check. Returns whether Ollama is reachable and responsive.
    """
    api_result = _api_request("GET", "/api/tags", timeout=ollama_config.health_check_timeout)
    if api_result.get("success"):
        return {
            "healthy": True,
            "method": "api",
            "status_code": api_result.get("status_code"),
            "model_count": len(api_result.get("data", []) or []),
        }

    # Try SSH
    ssh_result = _ssh_run("ollama list 2>/dev/null | wc -l", timeout=10)
    if ssh_result["success"]:
        try:
            model_count = int(ssh_result["stdout"].strip())
            return {
                "healthy": True,
                "method": "ssh",
                "model_count": max(0, model_count - 1),  # Subtract header line
            }
        except ValueError:
            pass

    return {
        "healthy": False,
        "method": ollama_config.control_method,
        "error": "Ollama server not reachable via API or SSH",
    }


def get_model_details(model_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    status = get_status()
    for m in status.get("models", []):
        if m["name"] == model_name:
            return {
                **m,
                "server_running": status.get("server_running"),
                "control_method": status.get("control_method"),
                "version": status.get("version"),
            }
    return {
        "name": model_name,
        "found": False,
        "server_running": status.get("server_running"),
    }


# ── Dashboard API endpoints ─────────────────────────────────────────────────────

def register_ollama_endpoints(app: Any):
    """
    Register Ollama control endpoints on the FastAPI dashboard app.
    Call this from the dashboard app's startup.
    """
    from fastapi import HTTPException

    @app.get("/api/ollama/status")
    def ollama_status():
        """Get detailed Ollama server status."""
        return get_status()

    @app.get("/api/ollama/health")
    def ollama_health():
        """Quick health check."""
        return check_health()

    @app.post("/api/ollama/start")
    def ollama_start():
        """Start the Ollama server."""
        return start_server()

    @app.post("/api/ollama/stop")
    def ollama_stop():
        """Stop the Ollama server."""
        return stop_server()

    @app.post("/api/ollama/restart")
    def ollama_restart():
        """Restart the Ollama server."""
        return restart_server()

    @app.get("/api/ollama/models")
    def ollama_models():
        """List all models."""
        return list_models()

    @app.post("/api/ollama/pull")
    def ollama_pull(payload: Dict[str, str]):
        """Pull a model. Body: { "model": "qwen2.5:3b" }"""
        model = payload.get("model", ollama_config.default_model)
        return pull_model(model)

    @app.get("/api/ollama/model/{model_name}")
    def ollama_model_detail(model_name: str):
        """Get details for a specific model."""
        return get_model_details(model_name)

    @app.post("/api/ollama/config")
    def ollama_config_update(payload: Dict[str, Any]):
        """Update Ollama control configuration."""
        if "control_method" in payload:
            set_control_method(payload["control_method"])
        if "host" in payload:
            ollama_config.host = payload["host"]
        if "port" in payload:
            ollama_config.port = int(payload["port"])
        if "ssh_user" in payload:
            ollama_config.ssh_user = payload["ssh_user"]
        if "default_model" in payload:
            ollama_config.default_model = payload["default_model"]
        if "use_ssh" in payload:
            ollama_config.use_ssh = bool(payload["use_ssh"])
        return {
            "updated": payload,
            "current": {
                "host": ollama_config.host,
                "port": ollama_config.port,
                "control_method": ollama_config.control_method,
                "ssh_user": ollama_config.ssh_user,
                "default_model": ollama_config.default_model,
                "use_ssh": ollama_config.use_ssh,
            }
        }

    @app.get("/api/ollama/config")
    def ollama_config_get():
        """Get current Ollama control configuration."""
        return {
            "host": ollama_config.host,
            "port": ollama_config.port,
            "ssh_user": ollama_config.ssh_user,
            "ssh_host": ollama_config.ssh_host,
            "ssh_key": ollama_config.ssh_key,
            "default_model": ollama_config.default_model,
            "models_dir": ollama_config.models_dir,
            "control_method": ollama_config.control_method,
            "use_ssh": ollama_config.use_ssh,
            "pull_timeout": ollama_config.pull_timeout,
            "health_check_timeout": ollama_config.health_check_timeout,
        }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 -m token_optimiser.dashboard.ollama_control <command> [args]")
        print("Commands: status, health, start, stop, restart, pull <model>, list, details <model>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "status":
        result = get_status()
        print(json.dumps(result, indent=2))
    elif command == "health":
        result = check_health()
        print(json.dumps(result, indent=2))
    elif command == "start":
        result = start_server()
        print(json.dumps(result, indent=2))
    elif command == "stop":
        result = stop_server()
        print(json.dumps(result, indent=2))
    elif command == "restart":
        result = restart_server()
        print(json.dumps(result, indent=2))
    elif command == "pull":
        if len(sys.argv) < 3:
            model = ollama_config.default_model
            print(f"No model specified, using default: {model}")
        else:
            model = sys.argv[2]
        result = pull_model(model)
        print(json.dumps(result, indent=2))
    elif command == "list":
        result = list_models()
        print(json.dumps(result, indent=2))
    elif command == "details":
        if len(sys.argv) < 3:
            print("Usage: ollama_control details <model_name>")
            sys.exit(1)
        result = get_model_details(sys.argv[2])
        print(json.dumps(result, indent=2))
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

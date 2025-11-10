import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
# Enable CORS for frontend - single, consistent configuration
# No credentials, so we can use "*" for origins
CORS(app, resources={r"/api/*": {
    "origins": "*",  # or ["http://localhost:5173"] if you need specific origin
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "supports_credentials": False
}})

# Get the ai-agents directory
AI_AGENTS_DIR = Path(__file__).parent.absolute()
BANKING_CORPUS_DIR = AI_AGENTS_DIR / "data" / "banking_corpus"
RESULTS_DIR = AI_AGENTS_DIR / "data" / "results"

# Ensure directories exist
BANKING_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/run-orchestrator', methods=['POST'])
def run_orchestrator():
    """Run the orchestrator with a prompt"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        print(f"\n🔵 API: Running orchestrator with prompt: {prompt[:100]}...")
        
        # Change to ai-agents directory
        original_cwd = os.getcwd()
        os.chdir(AI_AGENTS_DIR)
        
        try:
            # Run orchestrator
            result = subprocess.run(
                ['python', '-m', 'agents.orchestrator', '--ask', prompt],
                capture_output=True,
                text=True,
                cwd=str(AI_AGENTS_DIR),
                timeout=300  # 5 minute timeout
            )
            
            # Restore original directory
            os.chdir(original_cwd)
            
            # Parse output
            output = result.stdout
            error_output = result.stderr
            
            print(f"🔵 API: Orchestrator stdout length: {len(output)}")
            if error_output:
                print(f"🔵 API: Orchestrator stderr: {error_output[:200]}")
            
            # Try to extract JSON sections from output
            plan = {}
            summary = ""
            links = {}
            errors = None
            
            if "=== PLAN ===" in output:
                plan_start = output.find("=== PLAN ===")
                summary_start = output.find("=== SUMMARY ===")
                if summary_start > plan_start:
                    plan_text = output[plan_start+12:summary_start].strip()
                    try:
                        plan = json.loads(plan_text)
                    except:
                        pass
        
            if "=== SUMMARY ===" in output:
                summary_start = output.find("=== SUMMARY ===")
                links_start = output.find("=== LINKS ===")
                if links_start > summary_start:
                    summary = output[summary_start+15:links_start].strip()
            
            if "=== LINKS ===" in output:
                links_start = output.find("=== LINKS ===")
                errors_start = output.find("=== ERRORS ===")
                if errors_start > links_start:
                    links_text = output[links_start+13:errors_start].strip()
                    try:
                        links = json.loads(links_text)
                    except:
                        pass
            
            if "=== ERRORS ===" in output:
                errors_start = output.find("=== ERRORS ===")
                errors_text = output[errors_start+14:].strip()
                try:
                    errors = json.loads(errors_text)
                except:
                    errors = errors_text if errors_text != "null" else None
            
            # Check for metrics chart
            metrics_chart_path = RESULTS_DIR / "metrics_chart.png"
            has_metrics = metrics_chart_path.exists()
            
            # Determine intent from plan
            intent = plan.get('intent', '').lower() if isinstance(plan, dict) else ''
            
            print(f"🔵 API: Extracted intent: {intent}, has_metrics: {has_metrics}")
            
            return jsonify({
                'success': True,
                'plan': plan,
                'summary': summary,
                'links': links,
                'errors': errors,
                'intent': intent,
                'has_metrics': has_metrics,
                'stdout': output,
                'stderr': error_output
            })
        except subprocess.TimeoutExpired:
            os.chdir(original_cwd)
            return jsonify({'error': 'Orchestrator timed out after 5 minutes'}), 500
        except Exception as e:
            os.chdir(original_cwd)
            print(f"🔴 API: Error running orchestrator: {e}")
            return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        print(f"🔴 API: Error in run_orchestrator: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-file', methods=['POST'])
def upload_file():
    """Upload file to banking_corpus and build RAG index"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save file to banking_corpus
            filename = secure_filename(file.filename)
            # Convert PDF/images to .txt if needed, or save as-is
            if filename.endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                # For now, save as .txt (you may want to add PDF/image extraction)
                txt_filename = filename.rsplit('.', 1)[0] + '.txt'
                filepath = BANKING_CORPUS_DIR / txt_filename
            else:
                filepath = BANKING_CORPUS_DIR / filename
            
            file.save(str(filepath))
            
            # Build RAG index
            os.chdir(AI_AGENTS_DIR)
            result = subprocess.run(
                ['python', 'scripts/build_rag_index.py'],
                capture_output=True,
                text=True,
                cwd=str(AI_AGENTS_DIR)
            )
            
            return jsonify({
                'success': True,
                'filename': filename,
                'message': 'File uploaded and indexed successfully',
                'stdout': result.stdout,
                'stderr': result.stderr
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics-chart', methods=['GET'])
def get_metrics_chart():
    """Get the metrics chart PNG file"""
    chart_path = RESULTS_DIR / "metrics_chart.png"
    if chart_path.exists():
        return send_file(str(chart_path), mimetype='image/png')
    else:
        return jsonify({'error': 'Metrics chart not found'}), 404

# CORS is handled by Flask-CORS, no need for after_request handler
# Removing to avoid duplicate headers

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    # Use port 5001 instead of 5000 (5000 is often used by macOS AirPlay Receiver)
    app.run(port=5001, debug=True, host='127.0.0.1')
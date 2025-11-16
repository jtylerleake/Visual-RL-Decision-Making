"""
Modern web-based GUI for launching experiments.
Uses Flask and SocketIO for real-time updates.
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
from pathlib import Path
from gui.experiment_runner import ExperimentRunner

# Get the directory where this file is located
gui_dir = Path(__file__).parent
template_dir = gui_dir / 'templates'
static_dir = gui_dir / 'static'

app = Flask(__name__, 
            template_folder=str(template_dir), 
            static_folder=str(static_dir))
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# Initialize experiment runner
runner = ExperimentRunner()

# Store running experiments
running_experiments = {}


@app.route('/')
def index():
    """Serve the main GUI page"""
    return render_template('index.html')


@app.route('/api/experiments', methods=['GET'])
def get_experiments():
    """Get list of available experiments"""
    try:
        experiments = runner.available_experiments
        return jsonify({'success': True, 'experiments': experiments})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/run', methods=['POST'])
def run_experiment():
    """Start running an experiment"""
    try:
        data = request.json
        experiment_name = data.get('experiment_name')
        
        if not experiment_name:
            return jsonify({'success': False, 'error': 'No experiment specified'}), 400
        
        if experiment_name not in runner.available_experiments:
            return jsonify({'success': False, 'error': 'Unknown experiment'}), 400
        
        # Check if already running
        if experiment_name in running_experiments:
            return jsonify({'success': False, 'error': 'Experiment already running'}), 400
        
        # Start experiment in background thread
        thread = threading.Thread(
            target=_run_experiment_thread,
            args=(experiment_name,),
            daemon=True
        )
        thread.start()
        running_experiments[experiment_name] = {'thread': thread, 'status': 'running'}
        
        return jsonify({'success': True, 'message': 'Experiment started'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def _run_experiment_thread(experiment_name):
    """Run experiment in background thread with progress updates"""
    try:
        # Emit progress updates via SocketIO
        socketio.emit('progress', {
            'experiment': experiment_name,
            'progress': 0,
            'status': 'Starting experiment...'
        })
        
        socketio.emit('progress', {
            'experiment': experiment_name,
            'progress': 25,
            'status': 'Loading configuration...'
        })
        
        socketio.emit('progress', {
            'experiment': experiment_name,
            'progress': 50,
            'status': 'Running experiment...'
        })
        
        # Run the experiment
        results = runner.run_experiment(experiment_name=experiment_name)
        
        socketio.emit('progress', {
            'experiment': experiment_name,
            'progress': 75,
            'status': 'Processing results...'
        })
        
        # Generate summary
        summary = runner.get_experiment_summary(results)
        
        socketio.emit('progress', {
            'experiment': experiment_name,
            'progress': 100,
            'status': 'Experiment completed'
        })
        
        # Emit results
        socketio.emit('experiment_complete', {
            'experiment': experiment_name,
            'results': results,
            'summary': summary
        })
        
    except Exception as e:
        socketio.emit('experiment_error', {
            'experiment': experiment_name,
            'error': str(e)
        })
    finally:
        # Remove from running experiments
        if experiment_name in running_experiments:
            del running_experiments[experiment_name]


@app.route('/api/results', methods=['GET'])
def get_results():
    """Get results for a specific experiment (if stored)"""
    # This could be extended to retrieve saved results from files
    return jsonify({'success': True, 'message': 'Results endpoint'})


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'message': 'Connected to experiment server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    pass


def run(host='127.0.0.1', port=5000, debug=False):
    """Start the Flask server"""
    print(f"Starting Experiment Launcher GUI at http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)

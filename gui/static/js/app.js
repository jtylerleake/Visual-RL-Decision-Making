/**
 * Modern JavaScript for Experiment Launcher GUI
 * Handles all frontend interactions and real-time updates
 */

// Initialize Socket.IO connection
const socket = io();

// Global state
let currentResults = null;
let resultsChart = null;

// DOM Elements
const experimentSelect = document.getElementById('experiment-select');
const refreshBtn = document.getElementById('refresh-btn');
const runBtn = document.getElementById('run-btn');
const clearBtn = document.getElementById('clear-btn');
const progressBar = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const statusText = document.getElementById('status-text');
const summaryContent = document.getElementById('summary-content');
const plotsContent = document.getElementById('plots-content');
const summaryTab = document.getElementById('summary-tab');
const plotsTab = document.getElementById('plots-tab');
const tabButtons = document.querySelectorAll('.tab-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toast-message');

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadExperiments();
    setupEventListeners();
    setupSocketListeners();
});

/**
 * Setup event listeners for UI interactions
 */
function setupEventListeners() {
    // Refresh button
    refreshBtn.addEventListener('click', () => {
        loadExperiments();
        showToast('Experiments refreshed', 'success');
    });

    // Run button
    runBtn.addEventListener('click', () => {
        const experiment = experimentSelect.value;
        if (!experiment) {
            showToast('Please select an experiment', 'error');
            return;
        }
        runExperiment(experiment);
    });

    // Clear button
    clearBtn.addEventListener('click', () => {
        clearResults();
    });

    // Tab switching
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            switchTab(tabName);
        });
    });
}

/**
 * Setup Socket.IO event listeners for real-time updates
 */
function setupSocketListeners() {
    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('connected', (data) => {
        console.log('Server message:', data.message);
    });

    socket.on('progress', (data) => {
        updateProgress(data.progress, data.status);
    });

    socket.on('experiment_complete', (data) => {
        handleExperimentComplete(data);
    });

    socket.on('experiment_error', (data) => {
        handleExperimentError(data);
    });
}

/**
 * Load available experiments from API
 */
async function loadExperiments() {
    try {
        showLoading(true);
        const response = await fetch('/api/experiments');
        const data = await response.json();

        if (data.success) {
            experimentSelect.innerHTML = '';
            
            if (data.experiments.length === 0) {
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'No experiments available';
                experimentSelect.appendChild(option);
            } else {
                data.experiments.forEach(exp => {
                    const option = document.createElement('option');
                    option.value = exp;
                    option.textContent = exp;
                    experimentSelect.appendChild(option);
                });
            }
        } else {
            showToast('Failed to load experiments: ' + data.error, 'error');
        }
    } catch (error) {
        showToast('Error loading experiments: ' + error.message, 'error');
        console.error('Error:', error);
    } finally {
        showLoading(false);
    }
}

/**
 * Run an experiment
 */
async function runExperiment(experimentName) {
    try {
        runBtn.disabled = true;
        runBtn.innerHTML = 'Running...';
        updateProgress(0, 'Starting experiment...');
        clearResults();

        const response = await fetch('/api/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ experiment_name: experimentName })
        });

        const data = await response.json();

        if (data.success) {
            showToast('Experiment started successfully', 'success');
        } else {
            showToast('Failed to start experiment: ' + data.error, 'error');
            runBtn.disabled = false;
            runBtn.innerHTML = 'Run Experiment';
        }
    } catch (error) {
        showToast('Error starting experiment: ' + error.message, 'error');
        runBtn.disabled = false;
        runBtn.innerHTML = 'Run Experiment';
        console.error('Error:', error);
    }
}

/**
 * Update progress bar and status
 */
function updateProgress(progress, status) {
    progressBar.style.width = `${progress}%`;
    progressText.textContent = `${Math.round(progress)}%`;
    statusText.textContent = status || 'Processing...';
}

/**
 * Handle experiment completion
 */
function handleExperimentComplete(data) {
    currentResults = data.results;
    
    // Update summary - remove empty state if present
    if (data.summary) {
        const emptyState = summaryContent.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }
        summaryContent.textContent = data.summary;
        summaryContent.className = 'summary-content';
    }

    // Update plots
    if (data.results) {
        createResultsChart(data.results);
    }

    // Reset UI
    runBtn.disabled = false;
    runBtn.innerHTML = 'Run Experiment';
    updateProgress(100, 'Experiment completed');
    showToast('Experiment completed successfully!', 'success');

    // Switch to summary tab
    switchTab('summary');
}

/**
 * Handle experiment error
 */
function handleExperimentError(data) {
    runBtn.disabled = false;
    runBtn.innerHTML = 'Run Experiment';
    updateProgress(0, 'Experiment failed');
    showToast('Experiment failed: ' + data.error, 'error');
    
    if (summaryContent) {
        summaryContent.textContent = `Error: ${data.error}`;
    }
}

/**
 * Create results chart using Chart.js
 */
function createResultsChart(results) {
    const ctx = document.getElementById('results-chart');
    if (!ctx) return;

    // Destroy existing chart
    if (resultsChart) {
        resultsChart.destroy();
    }

    // Clear empty state
    const emptyState = plotsContent.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }

    // Prepare chart data
    let labels, dataValues, colors;

    if (results.success) {
        labels = ['Success', 'Duration (s)', 'Config Loaded'];
        dataValues = [
            results.success ? 1 : 0,
            results.duration_seconds || 0,
            1
        ];
        colors = [
            'rgba(39, 174, 96, 0.85)',   // Success green
            'rgba(52, 152, 219, 0.85)',  // Academic blue
            'rgba(44, 62, 80, 0.85)'     // Professional dark
        ];
    } else {
        labels = ['Status'];
        dataValues = [0];
        colors = ['rgba(231, 76, 60, 0.85)'];  // Error red
    }

    // Create new chart
    resultsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Experiment Results',
                data: dataValues,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.85', '1')),
                borderWidth: 1.5,
                borderRadius: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: `Experiment: ${results.experiment_name || 'Results'}`,
                    color: '#212529',
                    font: {
                        size: 16,
                        weight: '600',
                        family: "'Roboto Slab', serif"
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(255, 255, 255, 0.98)',
                    titleColor: '#212529',
                    bodyColor: '#495057',
                    borderColor: '#dee2e6',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    boxPadding: 6
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#6c757d',
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(222, 226, 230, 0.8)'
                    }
                },
                x: {
                    ticks: {
                        color: '#6c757d',
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(222, 226, 230, 0.8)'
                    }
                }
            }
        }
    });
}

/**
 * Clear results display
 */
function clearResults() {
    currentResults = null;
    
    // Clear summary - restore empty state
    summaryContent.innerHTML = `
        <div class="empty-state">
            <p>No results available. Run an experiment to display results.</p>
        </div>
    `;
    summaryContent.className = 'summary-content';

    // Clear chart
    if (resultsChart) {
        resultsChart.destroy();
        resultsChart = null;
    }

    // Restore empty state for plots
    plotsContent.innerHTML = `
        <div class="empty-state">
            <p>No visualizations available. Run an experiment to generate plots.</p>
        </div>
        <canvas id="results-chart"></canvas>
    `;

    // Reset progress
    updateProgress(0, 'Ready');
    statusText.textContent = 'Results cleared';
    showToast('Results cleared', 'success');
}

/**
 * Switch between tabs
 */
function switchTab(tabName) {
    // Update tab buttons
    tabButtons.forEach(btn => {
        if (btn.dataset.tab === tabName) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Update tab content
    if (tabName === 'summary') {
        summaryTab.classList.add('active');
        plotsTab.classList.remove('active');
    } else if (tabName === 'plots') {
        summaryTab.classList.remove('active');
        plotsTab.classList.add('active');
    }
}

/**
 * Show/hide loading overlay
 */
function showLoading(show) {
    if (show) {
        loadingOverlay.classList.remove('hidden');
    } else {
        loadingOverlay.classList.add('hidden');
    }
}

/**
 * Show toast notification
 */
function showToast(message, type = 'success') {
    toastMessage.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            toast.classList.add('hidden');
        }, 300);
    }, 3000);
}


let mediaRecorder;
let audioChunks = [];
const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const status = document.getElementById('status');
const visualizer = document.getElementById('visualizer');
const results = document.getElementById('results');
const transcriptionText = document.getElementById('transcriptionText');
const threatResult = document.getElementById('threatResult');
let siriWave;

// Create waveform visualizer for audio recording
function createWaveform() {
    visualizer.innerHTML = '';

    siriWave = new SiriWave({
        container: visualizer,
        width: visualizer.offsetWidth,
        height: 200,
        style: 'ios9',
        speed: 0.08,
        amplitude: 1,
        color: '#4CAF50',
        autostart: true
    });
}


// Event listener for the record button
recordButton.addEventListener('click', async () => {
    try {
        // Request access to user's microphone
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        // Handle data availability during recording
        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        // Handle stop event for processing audio
        mediaRecorder.onstop = async () => {
            if (siriWave) siriWave.stop(); // Stop the waveform animation
            visualizer.classList.remove('active');
            status.textContent = "Processing audio...";
            await processAudio();
        };
        

        // Start recording
        mediaRecorder.start(100); // Record in chunks of 100ms

        // UI updates during recording
        recordButton.disabled = true;
        stopButton.disabled = false;
        status.textContent = "Recording... Speak now.";
        visualizer.classList.add('active');
        createWaveform();
        results.style.display = 'none';

    } catch (error) {
        console.error('Error accessing microphone:', error);
        status.textContent = "Error accessing microphone. Please check permissions.";
    }
});

// Event listener for the stop button
stopButton.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        stopButton.disabled = true;
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
});

// Function to process the recorded audio and send it to the backend for analysis
async function processAudio() {
    try {
        threatResult.innerHTML = '<div class="loading"></div> Analyzing...';
        results.style.display = 'block';

        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');

        // Send audio to the backend for analysis
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        // Debugging: Log the result object to check if transcription exists
        console.log('Result from server:', result);

        // Display transcription or error message
        if (result.transcription) {
            transcriptionText.textContent = result.transcription;
        } else {
            transcriptionText.textContent = "Could not transcribe audio.";
        }

        // Display threat status based on analysis
        if (result.is_threat) {
            threatResult.className = 'threat-indicator threat';
            threatResult.innerHTML = `🚨 ${result.threat_status}`;
        } else {
            threatResult.className = 'threat-indicator no-threat';
            threatResult.innerHTML = `✅ ${result.threat_status}`;
        }

        status.textContent = "Analysis complete. Ready to record again.";

    } catch (error) {
        console.error('Error:', error);
        transcriptionText.textContent = "Error processing audio.";
        threatResult.className = 'threat-indicator threat';
        threatResult.textContent = "⚠️ Analysis failed.";
        status.textContent = "Error occurred during analysis. Please try again.";
    }
}

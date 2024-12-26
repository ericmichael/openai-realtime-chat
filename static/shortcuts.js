let isRecording = false;

function shortcuts(e) {
    // Skip if we're in an input or textarea
    if (['input', 'textarea'].includes(e.target.tagName.toLowerCase())) {
        return;
    }

    if (e.code === 'Space' && e.type === 'keydown' && !isRecording) {
        e.preventDefault();
        isRecording = true;
        const recordButton = document.querySelector('.record-button');
        if (recordButton && !recordButton.disabled) {
            recordButton.click();
        }
    } else if (e.code === 'Space' && e.type === 'keyup' && isRecording) {
        e.preventDefault();
        isRecording = false;
        const stopButton = document.querySelector('.stop-button');
        if (stopButton && !stopButton.disabled) {
            stopButton.click();
        }
    }
}

document.addEventListener('keydown', shortcuts, false);
document.addEventListener('keyup', shortcuts, false); 
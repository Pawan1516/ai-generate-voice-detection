// DOM Elements
const detectBtn = document.getElementById('detectBtn');
const langSel = document.getElementById('languageSelect');
const apiKeyInput = document.getElementById('apiKeyInput');

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileNameDisplay = document.getElementById('fileName');
const removeFileBtn = document.getElementById('removeFileBtn');

const audioPlayback = document.getElementById('audioPlayback');
const previewContainer = document.getElementById('previewContainer');
const resultOverlay = document.getElementById('resultOverlay');

// Global State
let uploadedFileBase64 = null;

// Pre-fill for demo (Optional)
apiKeyInput.value = "700d33b8cc4f1bb2fc4171ec7192e2a0c629b2400484d3c472b727b01a762c11";

// --- File Upload Logic ---
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

async function handleFile(file) {
    if (!file.name.toLowerCase().endsWith('.mp3')) {
        showError("Only MP3 files are supported.");
        return;
    }

    fileNameDisplay.textContent = file.name;
    dropZone.classList.add('hidden');
    fileInfo.classList.remove('hidden');

    // Convert to Base64
    try {
        uploadedFileBase64 = await fileToBase64(file);

        // Set Preview
        const url = URL.createObjectURL(file);
        audioPlayback.src = url;
        previewContainer.classList.remove('hidden');

        validationCheck();
    } catch (e) {
        showError("Failed to read file.");
    }
}

removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    uploadedFileBase64 = null;
    fileInput.value = '';
    fileInfo.classList.add('hidden');
    dropZone.classList.remove('hidden');
    audioPlayback.src = '';
    previewContainer.classList.add('hidden');
    validationCheck();
});

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = error => reject(error);
    });
}

// --- Detection Logic ---
langSel.addEventListener('change', validationCheck);
apiKeyInput.addEventListener('input', validationCheck);

detectBtn.addEventListener('click', async () => {
    detectBtn.classList.add('loading');
    detectBtn.disabled = true;
    detectBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';

    const apiKey = apiKeyInput.value.trim();

    if (!apiKey) {
        showError("Please enter your API Key.");
        resetBtn();
        return;
    }

    if (!uploadedFileBase64) {
        showError("No audio file selected.");
        resetBtn();
        return;
    }

    const payload = {
        language: langSel.value,
        audioFormat: 'mp3',
        audioBase64: uploadedFileBase64
    };

    try {
        const response = await fetch(window.API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': apiKey
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) throw new Error(data.message || 'Error');

        showResult(data);

    } catch (error) {
        showError(error.message);
    } finally {
        resetBtn();
    }
});

function resetBtn() {
    detectBtn.classList.remove('loading');
    detectBtn.disabled = false;
    detectBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Analyze Voice';
}

function validationCheck() {
    const hasLang = langSel.value !== "";
    const hasKey = apiKeyInput.value.trim() !== "";
    const hasAudio = !!uploadedFileBase64;

    // Enable button only if all fields are valid? 
    // Or let them click and show error? 
    // Let's enable if at least audio is there, to allow error msgs for others.
    // Actually, strict UI usually disables until valid.
    detectBtn.disabled = !(hasLang && hasAudio && hasKey);
}

// --- Results & Errors ---
function showResult(data) {
    const isHuman = data.classification === 'HUMAN';
    const percent = Math.round(data.confidenceScore * 100);

    const iconWrapper = document.getElementById('resultIcon');
    const title = document.getElementById('resultTitle');
    const fill = document.getElementById('confidenceFill');
    const val = document.getElementById('confidenceValue');
    const expl = document.getElementById('resultExplanation');

    if (isHuman) {
        iconWrapper.innerHTML = '<i class="fa-solid fa-user-check" style="color: var(--success)"></i>';
        title.innerText = 'Authentic Human Voice';
        title.style.color = 'var(--success)';
        fill.style.backgroundColor = 'var(--success)';
    } else {
        iconWrapper.innerHTML = '<i class="fa-solid fa-robot" style="color: var(--danger)"></i>';
        title.innerText = 'AI-Generated Voice';
        title.style.color = 'var(--danger)';
        fill.style.backgroundColor = 'var(--danger)';
    }

    val.innerText = percent + '% Confidence';
    expl.innerText = data.explanation;

    resultOverlay.classList.remove('hidden');
    setTimeout(() => {
        resultOverlay.classList.add('active');
        fill.style.width = percent + '%';
    }, 50);
}

window.closeResult = () => {
    resultOverlay.classList.remove('active');
    setTimeout(() => resultOverlay.classList.add('hidden'), 300);
};

window.showError = (msg) => {
    alert(msg);
};

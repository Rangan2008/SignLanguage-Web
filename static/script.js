const video = document.getElementById('video');
const outputImg = document.getElementById('output-img');
const currentSymbol = document.getElementById('current-symbol');
const word = document.getElementById('current-word');
const suggestions = [
  document.getElementById('suggest1'),
  document.getElementById('suggest2'),
  document.getElementById('suggest3'),
  document.getElementById('suggest4')
];

let wordOverridden = false;

async function setupWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  return new Promise(resolve => {
    video.onloadedmetadata = () => {
      resolve();
    };
  });
}

function captureFrame() {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg');
}

async function sendFrame() {
  if (video.videoWidth === 0 || video.videoHeight === 0) return;
  const image = captureFrame();
  const response = await fetch('/process_frame', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image })
  });
  const data = await response.json();

  if (data.processed_image) {
    outputImg.src = data.processed_image;
  }

  currentSymbol.textContent = data.current_symbol || '';

  if (!wordOverridden) {
    word.textContent = data.word || '';
  }

  suggestions.forEach((el, i) => {
    el.textContent = data['word' + (i + 1)] || '';
  });
}

async function loop() {
  await sendFrame();
  setTimeout(loop, 200);
}

function speakWord() {
  const text = word.textContent.trim();
  if (text) {
    const utterance = new SpeechSynthesisUtterance(text);
    speechSynthesis.speak(utterance);
  }
}

function clearWord() {
  word.textContent = '';
  wordOverridden = true;
}

suggestions.forEach(el => {
  el.addEventListener('click', () => {
    word.textContent = el.textContent;
    wordOverridden = true;
  });
});

setupWebcam().then(loop);

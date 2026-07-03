'use strict';
// Wire up the four demos once the DOM and TensorFlow.js are ready.

function bootstrap() {
  const err = document.getElementById('tf-error');
  if (typeof tf === 'undefined') {
    if (err) err.style.display = 'block';
    return;
  }
  initDemo1();
  initDemo2();
  initDemo3();
  initDemo4();
}

if (document.readyState === 'loading') {
  window.addEventListener('DOMContentLoaded', bootstrap);
} else {
  bootstrap();
}

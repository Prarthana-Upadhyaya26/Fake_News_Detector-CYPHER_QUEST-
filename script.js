let startX;
let startY;

function handleTouchStart(event) {
    startX = event.touches[0].clientX;
    startY = event.touches[0].clientY;
}

function handleTouchMove(event) {
    if (!startX || !startY) {
        return;
    }

    let endX = event.touches[0].clientX;
    let endY = event.touches[0].clientY;
    let diffX = startX - endX;
    let diffY = startY - endY;

    if (Math.abs(diffX) > Math.abs(diffY)) { // Detect horizontal swipe
        if (diffX > 50) {
            // Swipe left
            navigateToPage('left');
        } else if (diffX < -50) {
            // Swipe right
            navigateToPage('right');
        }
    }

    startX = null;
    startY = null;
}

function handleMouseDown(event) {
    startX = event.clientX;
    startY = event.clientY;
}

function handleMouseMove(event) {
    if (!startX || !startY) {
        return;
    }

    let endX = event.clientX;
    let endY = event.clientY;
    let diffX = startX - endX;
    let diffY = startY - endY;

    if (Math.abs(diffX) > Math.abs(diffY)) { // Detect horizontal swipe
        if (diffX > 50) {
            // Swipe left
            navigateToPage('left');
        } else if (diffX < -50) {
            // Swipe right
            navigateToPage('right');
        }
    }

    startX = null;
    startY = null;
}

function handleKeyDown(event) {
    if (event.key === 'ArrowLeft') {
        navigateToPage('left');
    } else if (event.key === 'ArrowRight') {
        navigateToPage('right');
    }
}

function navigateToPage(direction) {
    if (direction === 'left') {
        if (document.getElementById('homepage')) {
            window.location.href = 'homepage2.html';
        }
    } else if (direction === 'right') {
        if (document.getElementById('homepage2')) {
            window.location.href = 'homepage.html';
        }
    }
}

document.addEventListener('touchstart', handleTouchStart, false);
document.addEventListener('touchmove', handleTouchMove, false);
document.addEventListener('mousedown', handleMouseDown, false);
document.addEventListener('mousemove', handleMouseMove, false);
document.addEventListener('keydown', handleKeyDown, false);

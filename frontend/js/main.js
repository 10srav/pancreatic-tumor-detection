// Main JavaScript for Navigation and Common Logic

document.addEventListener('DOMContentLoaded', function () {
    // Highlight active nav link
    const currentLocation = location.href;
    const menuItem = document.querySelectorAll('.nav-link');
    const menuLength = menuItem.length;
    for (let i = 0; i < menuLength; i++) {
        if (menuItem[i].href === currentLocation) {
            menuItem[i].className += " active";
        }
    }
});

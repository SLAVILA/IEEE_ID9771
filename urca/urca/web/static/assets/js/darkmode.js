/*!
 * Dark Mode Switch v1.0.1 (https://github.com/coliff/dark-mode-switch)
 * Copyright 2021 C.Oliff
 * Licensed under MIT (https://github.com/coliff/dark-mode-switch/blob/main/LICENSE)
 */
var darkSwitch = document.getElementById("darkSwitch");
window.addEventListener("load", function () {
  if (darkSwitch) {
    initTheme();
    darkSwitch.addEventListener("change", function () {
      resetTheme();
    });
  }
});
function initTheme() {
  var darkThemeSelected =
    localStorage.getItem("darkSwitch") !== null &&
    localStorage.getItem("darkSwitch") === "dark";
  darkSwitch.checked = darkThemeSelected;
  darkThemeSelected
    ? document.documentElement.setAttribute("data-theme", "dark")
    : document.documentElement.removeAttribute("data-theme");
}
function resetTheme() {
  if (darkSwitch.checked) {
    document.documentElement.setAttribute("data-theme", "dark");
    localStorage.setItem("darkSwitch", "dark");
  } else {
    document.documentElement.removeAttribute("data-theme");
    localStorage.removeItem("darkSwitch");
  }
}
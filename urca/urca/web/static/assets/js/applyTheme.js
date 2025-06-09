var htmlElement = document.documentElement;

// Listen for changes to the data-theme attribute
const observer = new MutationObserver(function (mutations) {
  mutations.forEach(function (mutation) {
    if (mutation.attributeName === "data-theme") {
      // Theme has changed, update CSS filters
      const theme = htmlElement.getAttribute("data-theme");
      if (theme === "dark") {
        htmlElement.style.filter = "invert(1) hue-rotate(180deg)";
        // const images = document.querySelectorAll('html img');
        // images.forEach(function(img) {
        //     img.style.filter = 'invert(1) hue-rotate(180deg)';
        // });
      } else {
        htmlElement.style.filter = "none";
        const images = document.querySelectorAll("html img");
        images.forEach(function (img) {
          img.style.filter = "none";
        });
      }
    }
  });
});

// Start observing changes to the data-theme attribute
observer.observe(htmlElement, { attributes: true });

var darkThemeSelected =
  localStorage.getItem("darkSwitch") !== null &&
  localStorage.getItem("darkSwitch") === "dark";

darkThemeSelected
  ? document.documentElement.setAttribute("data-theme", "dark")
  : document.documentElement.removeAttribute("data-theme");

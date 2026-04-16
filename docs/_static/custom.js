document.addEventListener("DOMContentLoaded", function () {
  const path = window.location.pathname;

  if (path.includes("/user_guide/")) {
    document.body.classList.add("show-user-guide-sidebar");
  } else {
    document.body.classList.remove("show-user-guide-sidebar");
  }
});
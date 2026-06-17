/* RespGeomLib — minimal interactions: graceful figure placeholders + copy BibTeX */
(function () {
  "use strict";

  /* Replace missing figure images with a clean, neutral placeholder box.
     Drop the real files into assets/figures/ and the placeholders disappear. */
  document.querySelectorAll("img.paper-fig").forEach(function (img) {
    img.addEventListener("error", function () {
      if (img.dataset.failed) return;       // avoid loops
      img.dataset.failed = "1";
      var box = document.createElement("div");
      box.className = "fig-missing";
      var label = img.getAttribute("alt") || "Figure";
      box.innerHTML = "<span>" + label + "<br><small>(add image to assets/figures/)</small></span>";
      if (img.parentNode) img.parentNode.replaceChild(box, img);
    });
  });

  /* Copy BibTeX */
  var copyBtn = document.getElementById("copyBib");
  if (copyBtn) {
    copyBtn.addEventListener("click", function () {
      var text = document.getElementById("bibtex").innerText;
      function done() {
        copyBtn.classList.add("copied");
        copyBtn.textContent = "Copied";
        setTimeout(function () {
          copyBtn.classList.remove("copied");
          copyBtn.textContent = "Copy";
        }, 1600);
      }
      function fallback() {
        var ta = document.createElement("textarea");
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); done(); } catch (e) {}
        document.body.removeChild(ta);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(fallback);
      } else { fallback(); }
    });
  }

  /* The STL geometry preview is handled by airway-stl-viewer.mjs. */
})();

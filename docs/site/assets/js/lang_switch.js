// Rewrite the language selector links so that switching languages opens the
// corresponding page in the other language instead of the site index.
// The English site is deployed under /en/ and the Chinese site under /cn/,
// with parallel page paths. If a page has no counterpart in the target
// language, the link falls back to that language's index.
(function () {
  function rewrite() {
    var m = window.location.pathname.match(/^(.*\/)(en|cn)\/(.*)$/);
    if (!m) return;
    var prefix = m[1];
    var page = m[3];

    document.querySelectorAll(".md-header .md-select__link").forEach(function (a) {
      var lang = a.getAttribute("hreflang");
      if (!lang) return;
      var seg = lang === "zh" ? "cn" : lang;
      var target = prefix + seg + "/" + page;
      a.href = target;

      // If the corresponding page does not exist in the target language,
      // fall back to that language's index page.
      fetch(target, { method: "HEAD" }).then(function (resp) {
        if (!resp.ok) a.href = prefix + seg + "/";
      }).catch(function () {});
    });
  }

  document.addEventListener("DOMContentLoaded", rewrite);
  // Also run after instant navigations (Material's document$ observable).
  if (window.document$) window.document$.subscribe(rewrite);
})();

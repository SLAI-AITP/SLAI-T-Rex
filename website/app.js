(function () {
  "use strict";

  var languageToggle = document.getElementById("languageToggle");
  var copyCommand = document.getElementById("copyCommand");
  var defaultLanguage = "en";

  function setLanguage(language) {
    var elements = document.querySelectorAll("[data-en][data-zh]");
    var isChinese = language === "zh";

    document.documentElement.lang = isChinese ? "zh-CN" : "en";
    elements.forEach(function (element) {
      element.textContent = isChinese ? element.dataset.zh : element.dataset.en;
    });
    languageToggle.textContent = isChinese ? "English" : "中文";
    languageToggle.setAttribute(
      "aria-label",
      isChinese ? "Switch to English" : "切换到中文"
    );
    document.title = isChinese
      ? "SLAI T-Rex | Ascend SuperPOD 上的全参数后训练"
      : "SLAI T-Rex | Full-parameter post-training on Ascend SuperPOD";
    try {
      window.localStorage.setItem("slai-trex-language", language);
    } catch (error) {
      // Local storage may be unavailable in private browsing.
    }
  }

  if (languageToggle) {
    languageToggle.addEventListener("click", function () {
      var nextLanguage = document.documentElement.lang === "zh-CN" ? "en" : "zh";
      setLanguage(nextLanguage);
    });
  }

  if (copyCommand) {
    copyCommand.addEventListener("click", function () {
      var command = "docker pull quay.io/slai-t-rex/slai-t-rex:v1.0.0-a3-cann9.1.0";
      var copyPromise = navigator.clipboard
        ? navigator.clipboard.writeText(command)
        : Promise.reject(new Error("Clipboard API unavailable"));

      copyPromise.then(function () {
        var oldText = copyCommand.textContent;
        copyCommand.textContent = document.documentElement.lang === "zh-CN" ? "已复制" : "Copied";
        window.setTimeout(function () {
          copyCommand.textContent = oldText;
        }, 1800);
      }).catch(function () {
        copyCommand.textContent = document.documentElement.lang === "zh-CN" ? "请手动复制" : "Copy manually";
      });
    });
  }

  try {
    var savedLanguage = window.localStorage.getItem("slai-trex-language");
    if (savedLanguage === "zh" || savedLanguage === "en") {
      defaultLanguage = savedLanguage;
    }
  } catch (error) {
    // Use English when local storage is unavailable.
  }

  setLanguage(defaultLanguage);
}());

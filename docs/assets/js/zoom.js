// CREDITS TO https://www.cssscript.com/image-zoom-pan-hover-detail-view/
var addZoom = function (target) {
  // FETCH CONTAINER + IMAGE
  var container = document.getElementById(target),
      imgsrc = container.currentStyle || window.getComputedStyle(container, false),
      imgsrc = imgsrc.backgroundImage.slice(4, -1).replace(/"/g, ""),
      img = new Image();

  // LOAD IMAGE + ATTACH ZOOM
  img.src = imgsrc;
  img.onload = function () {
    var imgWidth = img.naturalWidth,
        imgHeight = img.naturalHeight,
        ratio = imgHeight / imgWidth,
        percentage = ratio * 100 + '%';

    // ZOOM ON MOUSE MOVE
    container.onmousemove = function (e) {
      var boxWidth = container.clientWidth,
          xPos = e.pageX - this.offsetLeft,
          yPos = e.pageY - this.offsetTop,
          xPercent = xPos / (boxWidth / 100) + '%',
          yPercent = yPos / (boxWidth * ratio / 100) + '%';

      Object.assign(container.style, {
        backgroundPosition: xPercent + ' ' + yPercent,
        backgroundSize: imgWidth + 'px'
      });
    };

    // RESET ON MOUSE LEAVE
    container.onmouseleave = function (e) {
      Object.assign(container.style, {
        backgroundPosition: 'center',
        backgroundSize: 'cover'
      });
    };
  }
};

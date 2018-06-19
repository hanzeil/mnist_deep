// 上传图片前预览
function previewImage(file) {
    var MAXWIDTH = 120;  // 最大图片宽度
    var MAXHEIGHT = 120;  // 最大图片高度
    if (file.files && file.files[0]) {
        var img = document.getElementById('preview');
        img.onload = function () {
            var rect = getZoomParam(MAXWIDTH, MAXHEIGHT, img.offsetWidth, img.offsetHeight);
            img.width = rect.width;
            img.height = rect.height;
        }
        var reader = new FileReader();
        reader.onload = function (evt) {
            img.src = evt.target.result;
        }
        reader.readAsDataURL(file.files[0]);
    } else {
        //兼容IE
        file.select();
        var src = document.selection.createRange().text;
        var img = document.getElementById('preview');
        img.filters.item('DXImageTransform.Microsoft.AlphaImageLoader').src = src;
    }
}

// 获取缩放的尺寸
function getZoomParam(maxWidth, maxHeight, width, height) {
    var param = {top: 0, left: 0, width: width, height: height};
    if (width > maxWidth || height > maxHeight) {
        rateWidth = width / maxWidth;
        rateHeight = height / maxHeight;
        if (rateWidth > rateHeight) {
            param.width = maxWidth;
            param.height = Math.round(height / rateWidth);
        } else {
            param.width = Math.round(width / rateHeight);
            param.height = maxHeight;
        }
    }
    param.left = Math.round((maxWidth - param.width) / 2);
    param.top = Math.round((maxHeight - param.height) / 2);
    return param;
}


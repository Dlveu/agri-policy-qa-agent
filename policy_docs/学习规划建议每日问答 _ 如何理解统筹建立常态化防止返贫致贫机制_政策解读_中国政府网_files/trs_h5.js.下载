var phoneWidth = parseInt(window.screen.width),
	_phoneWidth = document.documentElement.clientWidth,
	_isIpad = /iPad/.test(navigator.userAgent);

var htmlEleO = document.getElementsByTagName('html')[0];

var phoneScale = phoneWidth / 750;
var ua = navigator.userAgent;
if (/Android (\d+\.\d+)/.test(ua)) {
	var version = parseFloat(RegExp.$1);
	if (version > 2.3) {
		document.write('<meta name="viewport" content="width=750, minimum-scale = ' + phoneScale + ', maximum-scale = ' + phoneScale + ', target-densitydpi=device-dpi">')
	} else {
		document.write('<meta name="viewport" content="width=750, target-densitydpi=device-dpi">')
	}
}
if(htmlEleO.getAttribute('data-notmobile')){
	document.addEventListener('DOMContentLoaded',function(){
		var pageWidth = (document.getElementsByTagName('body')[0])&&document.getElementsByTagName('body')[0].clientWidth;
		var meta = document.createElement('meta');
		meta.setAttribute('name','viewport');
		meta.setAttribute('content','width='+pageWidth+', user-scalable=no, target-densitydpi=device-dpi');
		document.getElementsByTagName('head')[0].appendChild(meta);
	},false);
}else{
	if (_phoneWidth == 1194 && _isIpad) {
		document.write('<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" id="meta_pad_fun">')
	} else if (_phoneWidth == 1024 && _isIpad) {
		document.write('<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" id="meta_pad_fun">')
	} else if (phoneWidth > 992) {
		document.write('<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">')
	} else if (_phoneWidth == 768 && _isIpad) {
		document.write('<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" id="meta_pad_fun">')
	} else if (phoneWidth == 834 && _isIpad) {
		document.write('<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" id="meta_pad_fun">')
	} else {
		document.write('<meta name="viewport" content="width=750, user-scalable=no, target-densitydpi=device-dpi">')
	}
}



if(window.addEventListener){/*防止pc ie8 报错*/
window.addEventListener("onorientationchange" in window ? "orientationchange" : "resize", function() {
	var a = document.getElementById('meta_pad_fun');
	if (a != null) {
		if (phoneWidth == 834) {
			if (window.orientation === 180 || window.orientation === 0) {
				a.setAttribute('content', 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no')
			}
			if (window.orientation === 90 || window.orientation === -90) {
				a.setAttribute('content', 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no')
			}
		} else {
			if (window.orientation === 180 || window.orientation === 0) {
				a.setAttribute('content', 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no')
			}
			if (window.orientation === 90 || window.orientation === -90) {
				a.setAttribute('content', 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no')
			}
		}
	}
}, false);
}
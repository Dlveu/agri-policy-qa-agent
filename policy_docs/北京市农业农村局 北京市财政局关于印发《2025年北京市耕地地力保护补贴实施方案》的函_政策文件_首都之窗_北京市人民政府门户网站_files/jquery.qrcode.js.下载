(function( $ ){
	$.fn.qrcode = function(options) {
		var isHtml5 = false;
		if (document.getElementById("Canvas").getContext) {          
			isHtml5 = true;
		}
		// if options is string, 
		if( typeof options === 'string' ){
			options	= { text: options };
		}

		// set default values
		// typeNumber < 1 for automatic calculation
		options	= $.extend( {}, {
			render		: "canvas",
			width		: 256,
			height		: 256,
			typeNumber	: -1,
			correctLevel: QRErrorCorrectLevel.H,
			background : "#ffffff",
			foreground : "#000000"
		}, options);

		var set_logoImg = function(width,height,qrcode){
			var imgElement = $("#logoImg").attr("src");
			if(!imgElement){
				//create the img logo
				var img = $(document.createElement("IMG"))
					   .attr("src", options.src)
					   .attr("id", "logoImg")
					   .css(
					   		{
								"position" : "absolute",
					   			"z-Index" : 1000,
					   			"width" : width * 7 +"px",
					   			"height" : height * 7 + "px",
					   			"left" : ((qrcode.getModuleCount()-1)/2-3)*width+"px",
					   			"top": ((qrcode.getModuleCount()-1)/2-3)*height+"px"
					   		}
					   	).appendTo($('#qr_container'));
			}
		}
		var createCanvas	= function(){
			// create the qrcode itself
			var qrcode	= new QRCode(options.typeNumber, options.correctLevel);
			qrcode.addData(options.text);
			qrcode.make();

			// get canvas element
			var canvas	= $("Canvas").get(0);
			canvas.width	= options.width;
			canvas.height	= options.height;
			var ctx = canvas.getContext('2d');

			// compute tileW/tileH based on options.width/options.height
			var tileW	= options.width  / qrcode.getModuleCount();
			var tileH	= options.height / qrcode.getModuleCount();

			// draw in the canvas
			for( var row = 0; row < qrcode.getModuleCount(); row++ ){
				for( var col = 0; col < qrcode.getModuleCount(); col++ ){
					ctx.fillStyle = qrcode.isDark(row, col) ? options.foreground : options.background;
					var w = (Math.ceil((col+1)*tileW) - Math.floor(col*tileW));
					var h = (Math.ceil((row+1)*tileW) - Math.floor(row*tileW));
					ctx.fillRect(Math.round(col*tileW),Math.round(row*tileH), w, h);  
				}	
			}
			
			return canvas;
		}

		// from Jon-Carlos Rivera (https://github.com/imbcmdth)
		var createTable	= function(){
			// create the qrcode itself
			var qrcode	= new QRCode(options.typeNumber, options.correctLevel);
			qrcode.addData(options.text);
			qrcode.make();
			var $table;
			var tableTemp = $("#contentInfo").html();
			if(!tableTemp){
					// create table element
					$table = $('<table></table>')
					.css("width", options.width+"px")
					.css("height", options.height+"px")
					.css("border", "0px")
					.css("border-collapse", "collapse")
					.css('background-color', options.background)
					.attr('id',"contentInfo");
			}else{
				$("#contentInfo").html("");
				$table = $("#contentInfo");
			}
		  
			// compute tileS percentage
			var tileW	= options.width / qrcode.getModuleCount();
			var tileH	= options.height / qrcode.getModuleCount();

			// draw in the table
			for(var row = 0; row < qrcode.getModuleCount(); row++ ){
				var $row = $('<tr></tr>').css('height', tileH+"px").appendTo($table);
				
				for(var col = 0; col < qrcode.getModuleCount(); col++ ){
					$('<td></td>')
						.css('width', tileW+"px")
						.css('background-color', qrcode.isDark(row, col) ? options.foreground : options.background)
						.appendTo($row);
				}
			}
			
			return $table;
		}
  

		return this.each(function(){
			var element	=  isHtml5 ? createCanvas() : createTable();
			$(element).appendTo(this);
		});
	};
})( jQuery );

$(function () {
        var text = window.location.href;
        $('#div_div').qrcode({
          text: utf16to8(text),
          height: 132,
          width: 132,
          src: ''
        });
})
function utf16to8(str) { //转码
  var out, i, len, c;
  out = "";
  len = str.length;
  for (i = 0; i < len; i++) {
    c = str.charCodeAt(i);
    if ((c >= 0x0001) && (c <= 0x007F)) {
        out += str.charAt(i);
    } else if (c > 0x07FF) {
        out += String.fromCharCode(0xE0 | ((c >> 12) & 0x0F));
        out += String.fromCharCode(0x80 | ((c >> 6) & 0x3F));
        out += String.fromCharCode(0x80 | ((c >> 0) & 0x3F));
    } else {
        out += String.fromCharCode(0xC0 | ((c >> 6) & 0x1F));
        out += String.fromCharCode(0x80 | ((c >> 0) & 0x3F));
    }
  }
  return out;
}
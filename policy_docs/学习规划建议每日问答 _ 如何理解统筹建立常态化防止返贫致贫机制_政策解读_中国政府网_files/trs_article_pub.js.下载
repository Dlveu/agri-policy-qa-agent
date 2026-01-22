;$(function(){

	// 政策类文件主题分类处理
	$('.zcwj_ztfl').each(function (){
	  var t = $.trim($(this).text()) || '';
	  if (t.length >0) {
		$(this).text($(this).text().replace('主题分类\\', ''));
	  }
	});
	
	//字体大中小
  
	  // $(".index_switchsize .medium").css({
	  // 	color:'#ff0000'
	  // });
	  $(".index_switchsize span").click(function(){
  
		  // $(this).css({
		  // 	color:'#ff0000'
		  // }).siblings().css({
		  // 	color:'#666'
		  // });
  
		  //获取para的字体大小
		  var thisEle = $(".pages_content p,.pages_content,.pages_content font,.pages_content span,.pages_content div").css("font-size");
		  //parseFloat的第二个参数表示转化的进制，10就表示转为10进制
		  var textFontSize = parseFloat(thisEle , 10);
		  //javascript自带方法
		  var unit = thisEle.slice(-2); //获取单位
		  var cName = $(this).attr("class");
		  if(cName.indexOf('bigger') != -1){
				  textFontSize = 30;
			  $(this).addClass('on').siblings().removeClass('on')
		  }else if(cName.indexOf('big') != -1){
				  textFontSize = 25;
			  $(this).addClass('on').siblings().removeClass('on')
		  }
		  else if(cName.indexOf('default') != -1){
				//   textFontSize = 16;
				  location.reload();
			  $(this).addClass('on').siblings().removeClass('on')
		  }
		  //设置para的字体大小
		  $(".pages_content p,.pages_content,.pages_content font,.pages_content span,.pages_content div").css("font-size",  textFontSize + unit );
	  });
  
	  $(".index_switchsize .default").click(function(){
		  $(".pages_content p,.pages_content,.pages_content font,.pages_content span,.pages_content div").css("font-size","16px");
	  });
  
  
	//打印
	var printAreaCount = 0;
	var removePrintArea = function (id) {
	  $("iframe#" + id).remove();
	};
	$.fn.printArea = function () {
	  var ele = $(this);
	  var idPrefix = "printArea_";
	  removePrintArea(idPrefix + printAreaCount);
	  printAreaCount++;
	  var iframeId = idPrefix + printAreaCount;
	  var iframeStyle = 'position:absolute;width:0px;height:0px;left:-500px;top:-500px;';
	  iframe = document.createElement('IFRAME');
	  $(iframe).attr({
		style: iframeStyle,
		id: iframeId
	  });
	  document.body.appendChild(iframe);
	  var doc = iframe.contentWindow.document;
	  $(document).find("link").filter(function () {
		return $(this).attr("rel").toLowerCase() == "stylesheet" && $(this).attr('href').indexOf('mobile')<=0;
	  }).each(function () {
		doc.write('<link type="text/css" rel="stylesheet" href="' + $(this).attr("href") + '"/>');
	  });
	  $('style').each(function (){
		doc.write('<style>' + $(this).html() + '</style>');
	  });
	  doc.write('<div class="' + $(ele).attr("class") + '" style="width: ' + $(ele).width() + 'px">' + $(ele).html() + '</div>');
	  $('[data-print=js]').each(function (){
		doc.write('<script type="text/javascript" src="' + $(this).attr("src") + '"></script>');
	  });
	  doc.close();
	  var frameWindow = iframe.contentWindow;
	  frameWindow.close();
	  frameWindow.focus();
	  setTimeout(function (){
		frameWindow.print();
	  },500)
	}
	$(".customPrintIco").click(function(){
	  $(".printArea").printArea();
	});
  
  var list01li = $('.list01 li');
		var li_len = list01li.length;
		if(li_len == 0)
		$('.xg-list').hide();
  
  
	  $('.pages_content img').parent('span').css('text-indent',0)
  
  
  //鼠标滑过变透明
	  function HoverFade(butn){
		  $(butn).hover(
			  function(){
				  $(this).fadeTo(200,.8)
			  },
			  function(){
				  $(this).fadeTo(200,1)
			  }
		  )
	  }
	  HoverFade('.butn,.newPicture dl');
  
  
	  //回到顶部
	  $(".back_top").hide();//首先将.back_top隐藏
	  $(window).scroll(function() {
				  if ($(window).scrollTop() > 0) {$(".back_top").fadeIn(400);
				  } else { $(".back_top").fadeOut(400); }
	  });//当滚动条的位置处于距顶部100像素以下时，跳转链接出现，否则消失
	  $(".back_top a").click(function() {
				  $('body,html').animate({scrollTop: 0},200);
				  return false;
	  }); //当点击跳转链接后，回到页面顶部位置
  
  
  
  
		  //设置左侧最小高度
		  var leftHeight=$('.leftPart').height();
		  var rightHeight=$('.rightPart').height();
		  if(leftHeight<rightHeight){
			  $('.leftPart').height(rightHeight);
		  }
  
	  // 政策文件细览
	  if ($('.jiedu-blk').length > 0) {
		if ($('ul.jiedu_list li').length <= 0) {
		  $('.jiedu-blk').hide();
		}
	  }
  
  })
  
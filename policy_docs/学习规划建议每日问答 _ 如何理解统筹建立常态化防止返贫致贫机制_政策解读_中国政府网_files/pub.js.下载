/* 政府网站pub.js
* 1.IsIE8：是否ie8浏览器
* 2.IsPC：是否PC端
* 3.PcBackTop：PC端回到顶部
* 4.Debounce：防抖
* 6.MobiBackTop：手机端回到顶部
*/

var trs = (function () {
  function IsIE8() {
    var isIE = userAgentInfo.indexOf("compatible") > -1 && userAgentInfo.indexOf("MSIE") > -1;
    if (isIE) {
      var reIE = new RegExp("MSIE (\\d+\\.\\d+);");
      reIE.test(userAgentInfo);
      var fIEVersion = parseFloat(RegExp["$1"]);
      if (fIEVersion <= 8) {
        return;
      }
    }
  }
  function IsPC() {
    var userAgentInfo = navigator.userAgent;
    var Agents = ["Android", "iPhone", "SymbianOS", "Windows Phone", "iPad", "iPod"];
    var flag = true;
    for (var v = 0; v < Agents.length; v++) {
      if (userAgentInfo.indexOf(Agents[v]) > 0) {
        flag = false;
        break;
      }
    }
    return flag;
  }
  function debounce(fn,wait){
    var timer = null;
    return function(){
      if(timer !== null){
        clearTimeout(timer);
      }
      timer = setTimeout(fn,wait);
    }
  }
  function PcBackTop() {
    var dom = '<div id="back_top" class="back_top"><a>回到顶部</a></div>'
    $('#back_top').remove()
    if (IsPC()) {
      $('body').append(dom)
    }
    $(window).scroll(function () {
      if ($(window).scrollTop() > 0) {
        $("#back_top").fadeIn(400);
      } else { $("#back_top").fadeOut(400); }
    });
    $("#back_top").click(function () {
      $('body,html').animate({ scrollTop: 0 }, 200);
      return false;
    });
  }
  function MobiBackTop() {
    var dom = '<div id="mobi_back_top" class="m_back_top pchide"><a>回到顶部</a></div>'
    $('#mobi_back_top').remove()
    if (!IsPC()) {
      $('body').append(dom)
    }else{
      $('#mobi_back_top').remove()
    }
    $(window).scroll(function () {
      if ($(window).scrollTop() > 0) {
        $("#mobi_back_top").fadeIn(400);
      } else { $("#mobi_back_top").fadeOut(400); }
    });
    $("#mobi_back_top").click(function () {
      $('body,html').animate({ scrollTop: 0 },200);
      return false;
    });
  }
  function SearchFn(){
    $('#searchBtn').on('click',function(){
      var winWidth = $(window).width();
      $(this).toggleClass('active');
      $('.header .searchBox').toggleClass('active');
      if($(this).hasClass('active')){
        $('.search_mi').focus()
      }
      if ($("#iframe_header").length > 0 && $('.header >.pchide:visible').length >0) {
        $("#iframe_header_p",parent.document).height(Math.ceil($("#iframe_header").outerHeight()*winWidth/750));
      }
    })
  }
  function Tabs (dom) {
    var index = null
    $(dom+' .nav').on('mouseover','span',function(){
      index = $(this).index()
      $(this).addClass('active').siblings().removeClass('active');
      $(dom).find('.ctx .item').eq(index).addClass('active').siblings().removeClass('active');
    })
  }
  function H5Banner () {

  }

  return {
    IsPC: IsPC,
    IsIE8: IsIE8,
    PcBackTop: PcBackTop,
    MobiBackTop:MobiBackTop,
    SearchFn: SearchFn,
    Tabs: Tabs,
    H5Banner: H5Banner
  }
})();

/*** 注册登录等  ***/
function gettime() {
  var d = new Date();
  var formatNumber = function(num) {
    return num > 0 && num < 10 ? '0' + num : num;
  };
  return d.getFullYear() + "" + formatNumber(d.getMonth() + 1) + formatNumber(d.getDate()) + formatNumber(d.getHours()) + formatNumber(d.getMinutes()) + formatNumber(d.getSeconds());
}

function GetQueryString(name) {
  var reg = new RegExp("(^|&)"+ name +"=([^&]*)(&|$)");
  var r = decodeURI(window.location.search).substr(1).match(reg);
  if(r!=null)return  r[2]; return null;
}

var servicecode = GetQueryString("servicecode");
// var gourl = document.referrer;
var gourl = encodeURI(top.location.href);
var time = GetQueryString("time");
var sign = GetQueryString("sign");
var showname = GetQueryString("showname");
if(servicecode==null||servicecode==""||time==null||time==""||sign==null||sign==""){
  servicecode = "trs";
  time = gettime();
  sign = hex_md5(servicecode + 'CfQ9MR5u' + time);
}

$(function (){
  idm.config({
    debug: false, // 开启调试模式,调用的所有api的返回值会在客户端alert
    url:'https://user.www.gov.cn',		// 必填，idm地址
    servicecode: servicecode, 				// 必填，接入代码
    time: time,
    sign: sign,	// 必填，签名
    success:function(){	// 回调函数（成功）
      isloginNew();
    },
    fail:function(r)	{	// 回调函数(失败)
      $(".nologin").removeClass("hide");
    }
  });

  $('#userreg, #userreg1').click(function(){
    idm.user.openreg ({
      redirect:true,
      gourl: gourl,// 业务地址：非必填。回调总入口将以sp参数返回
      extargs: ''		// 扩展参数：非必填。回调将回传
    });
  });
  $('#userlogin, #userlogin1').click(function(){
    idm.user.openlogin ({
      redirect:true,
      gourl: gourl,// 业务地址：非必填。回调总入口将以sp参数返回
      extargs: ''		// 扩展参数：非必填。回调将回传
    });
  });
  $('#usercenter').click(function(){
    idm.user.openedit ({
      // gourl: gourl	// 票据：非必填。
    });
  });
  $('#showname1').click(function(){ // wap
    idm.user.open ({
      ticket: ''	// 票据：非必填。
    });
  });
  $("#logout, #logOutBtn1").click(function(){
    idm.user.logout({
      success:function(){	// 回调函数（成功）
        if(gourl!=null&&gourl!=""){
          top.location.href = gourl;
        }else{
          top.location.href = 'https://user.www.gov.cn';
        }
      },
      fail:function(r)	{	// 回调函数(失败)

      }
    });
  });
  $("#logOutBtn1").click(function(){ // wap
    idm.user.logout({
      success:function(){	// 回调函数（成功）
        var gourlM = GetQueryString("gourl");
        if(gourlM!=null&&gourlM!=""){
          top.location.href = gourlM;
        }else{
          top.location.href = 'https://user.www.gov.cn';
        }
      },
      fail:function(r)	{	// 回调函数(失败)

      }
    });
  });

  // 查看电脑版
  $('#godesktop').on('click', function () {
    var isShowPc = localStorage.setItem('isShowPc', true)
    $("meta[name='viewport']").attr("content", "width=1280, target-densitydpi=device-dpi")
    $('body').append('<div id="showMobi" class="showMobi"><img src="/images/trs_mobiIcon.png" ignoreapd="true" />返回手机版</div>')
  });
  // 返回手机版
  $('body').on('click', '#showMobi', function () {
    var isShowPc = localStorage.setItem('isShowPc', false)
    $("meta[name='viewport']").attr("content", "width=750, user-scalable=no, target-densitydpi=device-dpi")
    $('#showMobi').remove()
  });

});

function isloginNew(){
  idm.user.islogin ({
    success:function(r){	// 回调函数（成功）
      var showname = r.showname;
      if(showname==undefined){
        showname = "****";
      }
      //注意 这个脱敏手机号进行了修改 页面上显示只能达到八个字符
      if(showname.length==11){
        showname = showname.replace('****','*');
      }else{
        showname = "  "+showname;
      }
      $("#showname, #showname1").html(showname);
      $(".yeslogin").removeClass("hide");
      $(".nologin").addClass("hide");
    },
    fail:function(r)	{	// 回调函数(失败)
      $(".nologin").removeClass("hide");
      $(".yeslogin").addClass("hide");
    }
  });
}



function $Len(IdName){
  return document.getElementById(IdName).value.length;
}
function $SpaceLen(IdName){
  return document.getElementById(IdName).value.replace(/(^\s*)|(\s*$)/g , '').length;
}


function setTab(name,cursel,n){
	for(i=1;i<=n;i++){
		var menu=document.getElementById(name+i);
		var con=document.getElementById("con_"+name+"_"+i);
		
		menu.className=i==cursel?"hover":"";
		con.style.display=(i==cursel)?"block":"none";
	
	 }
}

function setTab1(name,cursel,n){
	for(i=1;i<=n;i++){
		var menu=document.getElementById(name+i);
		var con=document.getElementById("con_"+name+"_"+i);
		var con_a=document.getElementById(name+i+"_a");
		menu.className=i==cursel?"hover":"";
		con.style.display=i==cursel?"block":"none";
		con_a.style.display="none";
	 }
	 var con_a1=document.getElementById(name+cursel+"_a");
	 con_a1.style.display="block";
	 
}


$(function(){
	/*$(".footer2_1 ul li").not(".nobackground").mouseover(function(){
		$(this).addClass("asect").siblings().removeClass();	
		$(this).find(".select_content").show().parent().siblings().find(".select_content").hide();
	})
	$(".footer2_1 ul li").not(".nobackground").mouseout(function(){
		$(this).removeClass("asect")
		$(this).find(".select_content").hide();
	})*/
	$(".footer2_1>ul>li").mouseenter(function(){
		$(this).addClass("asect").find(".select_content").show();
	}).mouseleave(function () {
		$(this).removeClass("asect").find(".select_content").hide();	
	});
	
	$(".column3ri1_1 li").each(function(){
        $(".column3ri1_1 li").hover(function(){
            $(this).find("a").stop().animate({
                "background-position-y":0
            })
        },function(){
            $(this).find("a").stop().animate({
                "background-position-y":"5px"
            })
        })
    })
	
	$(".lmtit1 a").hover(function(){
		$(".spyp_main .spyp_con").eq($(this).index()).show().siblings().hide();	
	})
})
 
  
  



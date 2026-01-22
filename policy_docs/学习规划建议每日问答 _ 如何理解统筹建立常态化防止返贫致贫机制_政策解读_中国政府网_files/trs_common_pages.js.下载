// function toPage(dom,pageDom,num,content,type) {
//     var url = window.location.href
//     window.location.href = url.split('#')[0] + '#'+num
//     // 全文
//     if(type === 'alls'){
//         window.location.href = url.split('#')[0] + '#allContent'
//         dom.html(content.join('</trs_page_separator>'))
//         pageDom.hide()
//         return false
//     }
//     // 余下全文
//     if(type === 'all'){
//         if(content.slice(num-1).length>0){
//             dom.html((content.slice(num-1)).join('</trs_page_separator>'))
//             pageDom.hide()
//         }
//         return false
//     }
//     if (num-1 === 0) {
//         $('.prev').hide()
//     } else {
//         $('.prev').show()
//     }
//     if ((num-1) === (content.length) - 1) {
//         $('.next').hide()
//         $('.all').hide()
//     } else {
//         $('.next').show()
//         $('.all').show()
//     }
//     dom.html(content[num-1])
//     pageDom.find('.page').eq(num-1).addClass('on').siblings().removeClass('on')
// }

function simplePage(dom, pageDom, isShowPage) {
  isShowPage = isShowPage || false;
  if (!dom || dom.length <= 0) {
    if (!isShowPage) {
      $('#pagination').hide();
    }
    return;
  }
  var arr = dom.html().split('</trs_page_separator>');
  var pages = arr.length;
  if (pages === 1 || pages === 0) {
    if (!isShowPage) {
      $('#pagination').hide();
    }
    return;
  }

  var item = '';

  // if(pages<=simplePage.pageParamsO){
  //     for (var i = 1; i <= pages; i++) {
  //         item += '<a target="_self" class="page">' + i + '</a>'
  //     }
  // }else{
  //     for (var i = 1; i <= pages; i++) {
  //         item += '<a target="_self" class="page">' + i + '</a>'
  //     }
  // }

  // var head = '<p class="pageBtn">' +
  //     '<a target="_self" class="prev">&lt;</a>';

  // var bottom = '<a class="next" target="_self">&gt;</a>' +
  //     '<a class="all">余下全文</a>' +
  //     '<a class="alls">全文</a>' +
  //     '</p>'
  // pageDom.html(head + item + bottom)
  var index = window.location.href.split('#')[1] || 1;
  index = Number(index);
  $('#pagination').pagination({
    pageCount: pages,
    coping: true,
    activeCls: 'on',
    current: index,
    endPage: pages,
    count: 5,
    prevContent: '<',
    nextContent: '>',
    callback: function (api) {
      // var index = api.getCurrent();
      var _index = api.getCurrent();
      $('#pagination a[data-page=' + _index + ']').click();
      location.hash = '#' + _index;
      location.hash = '#' + _index;
      $('#UCAP-CONTENT .trs_paper_default').html(arr[_index - 1]);

      $('#pagination').append(
        '<font href="javascript:;" class="all" style="display:none !important;">余下全文</font>' +
          '<font href="javascript:;" class="alls">全文</font>',
      );
    },
  });
  $('#pagination').append(
    '<font href="javascript:;" class="all" style="display:none !important;">余下全文</font>' +
      '<font href="javascript:;" class="alls">全文</font>',
  );

  $(document).on('click', '#pagination .all', function () {
    $('#UCAP-CONTENT .trs_paper_default').html('');
    var index = Number($('#pagination .on').text()) - 1;
    for (var i = index; i < arr.length; i++) {
      $('#UCAP-CONTENT .trs_paper_default').append(arr[i]);
    }
    $('#pagination').hide();
  });
  $(document).on('click', '#pagination .alls', function () {
    $('#UCAP-CONTENT .trs_paper_default').html('');
    for (var i = 0; i < arr.length; i++) {
      $('#UCAP-CONTENT .trs_paper_default').append(arr[i]);
    }
    $('#pagination').hide();
  });

  // 初始化

  // if (index && index.length>0 && index<=pages) {
  //     $('#pagination a[data-page='+index+']').click();
  // }else {
  //     $('#pagination a[data-page=1]').click();
  // }

  $('#UCAP-CONTENT .trs_paper_default').html(arr[index - 1]);
  if (location.hash === '') {
    location.hash = '#1';
    location.hash = '#1';
  }
}

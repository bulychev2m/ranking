const x = document.getElementById('1')
      text = document.querySelector ('.text'),
      btn = document.getElementsByClassName ('button' )[0];

function check() {
    const a = +x.value

if (a > 0) {
    text.innerText = 'Это положительное число';
} else if (a < 0){
    text.innerText = 'Это отрицательное число';
} else {
    text.innerText = 'Это ноль';
}

}

btn.addEventListener('click', check);
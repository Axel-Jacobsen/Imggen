'use strict'

const COLOUR = {'red': 255, 'green': 255, 'blue': 255}

const resize_rgb_vec = () => {
		let mag = (COLOUR['red']**2 + COLOUR['green']**2 + COLOUR['blue']**2) ** (1/2)
		Object.keys(COLOUR).forEach(a => 
				COLOUR[a] *= 255 / mag
		)
}

const rgb_tuple = () => {
		return '(' + Math.round(COLOUR['red']) + ',' + Math.round(COLOUR['green']) + ',' + Math.round(COLOUR['blue']) + ')';
}

const html_rgb_string = () => {
		return 'rgb' + rgb_tuple()
}

const update_page = () => {
		let e1 = document.getElementById('colourSquare')
		e1.style.backgroundColor = html_rgb_string()

		let e2 = document.getElementById('rgb')
		e2.innerHTML = '(R,G,B) = ' + rgb_tuple()
}


document.onkeypress = e => {
		e = e || window.event
		switch (e.key) {
				case 'r':
						COLOUR['red'] += 10
						resize_rgb_vec(COLOUR)
						break
				case 'g':
						COLOUR['green'] += 10
						resize_rgb_vec(COLOUR)
						break
				case 'b':
						COLOUR['blue'] += 10
						resize_rgb_vec(COLOUR)
						break
				default:
						return
		}
		update_page()
}

console.log('Starting')
console.log("press the 'r', 'g', and 'b' keys")
resize_rgb_vec(COLOUR)
update_page()


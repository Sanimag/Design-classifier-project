// Canvas setup
const canvas = document.getElementById("canvas1");
const ctx = canvas.getContext("2d");
canvas.width = 800;
canvas.height = 500;

let score = 0;
let gameFrame = 0;
ctx.font = '50px Georgia';

// Mouse
let canvasPosition = canvas.getBoundingClientRect();
console.log(canvasPosition);

const mouse = {
	x: canvas.width/2,
	y: canvas.height/2,
	click: false
}

canvas.addEventListener('mousedown', function(event){
	mouse.x = event.x - canvasPosition.left;
	mouse.y = event.y - canvasPosition.top;
	mouse.click = true;
	console.log(mouse);
});
canvas.addEventListener('mouseup', function(event){
	mouse.click = false;
	console.log(mouse);
});
//Player

class Player {
	constructor(){
		this.x = canvas.width/2;
		this.y = canvas.height/2;
		this.radius = 50;
		this.angle = 0;
		this.frameX = 0;
		this.frameY = 0;
		this.frame = 0;
		this.spriteWidth = 498;
		this.spriteHeight = 327;
	}

	update(){
		const dx = this.x - mouse.x;
		const dy = this.y - mouse.y;

		if(mouse.x != this.x){
			this.x = this.x - dx/30;
		}
		if(mouse.y != this.y){
			this.y = this.y - dy/30;
		}
	}

	draw(){
		ctx.clearRect(0, 0, canvas.width, canvas.height)
		if(mouse.click){
			ctx.lineWidth = 0.2;
			ctx.beginPath();
			ctx.moveTo(this.x, this.y);
			ctx.lineTo(mouse.x, mouse.y);
			//ctx.stroke();
			ctx.closePath();
		}
		ctx.fillStyle = 'red';
		ctx.beginPath();
		ctx.arc(this.x, this.y, this.radius, 0, Math.PI*2);
		ctx.fill();
		ctx.closePath();
	}
}

class Bubble{
	
}

const player = new Player();

function animate(){
	player.update();
	player.draw();
	requestAnimationFrame(animate);
}
animate();

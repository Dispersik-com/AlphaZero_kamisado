const url = "http://" + window.location.host + '/handle';

async function sendMoveToServer(lastCell, newCell, piece) {

    const data = {
        last_cell: {
            row: lastCell.row,
            col: lastCell.col,
        },
        new_cell: {
            row: newCell.row,
            col: newCell.col,
        },
        piece: piece
    };

    const options = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    };

    try {
        const response = await fetch(url, options);

        if (!response.ok) {
            throw new Error('error HTTP: ' + response.status);
        }

        const responseData = await response.json();
        console.log('Response from server:', responseData);
        return responseData;
    } catch (error) {
        console.error('Error sending message:', error);
    }
}

class Monk {
    constructor(color, innerColor) {
        this.color = color;
        this.innerColor = innerColor;
    }
}

const colorBoard = [
    [7, 6, 5, 4, 3, 2, 1, 0],
    [2, 7, 4, 1, 6, 3, 0, 5],
    [1, 4, 7, 2, 5, 0, 3, 6],
    [4, 5, 6, 7, 0, 1, 2, 3],
    [3, 2, 1, 0, 7, 6, 5, 4],
    [6, 3, 0, 5, 2, 7, 4, 1],
    [5, 0, 3, 6, 1, 4, 7, 2],
    [0, 1, 2, 3, 4, 5, 6, 7]
];

const colorDict = {
    0: "brown",
    1: "green",
    2: "red",
    3: "yellow",
    4: "pink",
    5: "violet",
    6: "aqua",
    7: "orange"
};

let gameBoard = Array.from({ length: 8 }, () => Array.from({ length: 8 }));

let legal_moves = [];
for (let i = 1; i <= 6; i++) {
    for (let j = 0; j <= 7; j++) {
        legal_moves.push([i, j]);
    }
}

function createCircle(color, innerColor, imagePath) {
    const circle = document.createElement('div');
    circle.classList.add('circle');
    circle.style.backgroundColor = color;

    const innerCircle = document.createElement('div');
    innerCircle.classList.add('inner-circle');
    innerCircle.style.backgroundColor = innerColor;

//    const image = document.createElement('img');
//    image.src = "monk.png";
//    image.style.width = '100%';
//    image.style.height = '100%';
//    image.style.borderRadius = '50%';
//    innerCircle.appendChild(image);

    circle.appendChild(innerCircle);

    return circle;
}

let selectedCell = {
  piece: null,
  cell: null,
  row: -1,
  col: -1
};

function handleCellMouseOver(event) {
    if (selectedCell.cell !== null) {
        const cell = event.target;
        cell.style.borderWidth = '0.1vw';
        cell.style.borderColor = 'white';
    }
}

function handleCellMouseOut(event) {
    const cell = event.target;
    cell.style.borderWidth = '';
    cell.style.borderColor = '';
}

function handleCellClick(event) {
    let table = event.target.closest('table');
    const cell = event.target.closest('td');
    const rowIndex = cell.parentNode.rowIndex;
    const cellIndex = cell.cellIndex;

    if (gameBoard[rowIndex][cellIndex] instanceof Monk) {
        selectedCell.piece = gameBoard[rowIndex][cellIndex];
        selectedCell.cell = cell;
        selectedCell.row = rowIndex;
        selectedCell.col = cellIndex;
//        showLegalMoves(legal_moves);

    } else {
//        if (checkLegalMove(legal_moves, [rowIndex, cellIndex])) {
            if (selectedCell.piece !== null) {
                gameBoard[rowIndex][cellIndex] = selectedCell.piece;
                gameBoard[selectedCell.row][selectedCell.col] = null;

                const newCell = {row: rowIndex, col: cellIndex}

                sendMoveToServer(selectedCell, newCell, selectedCell.piece)
                      .then(response => {
                        console.log(response.legal_moves);
                        showLegalMoves(response.legal_moves);
                        if (response.winner){
                            alert(response.winner + " win!");
                            table.disable = true;
                         }
                      })
                      .catch(error => {
                        console.error(error);
                      });

                const circle = selectedCell.cell.querySelector('.circle');
                cell.appendChild(circle);
                selectedCell.cell = null;
                selectedCell.piece = null;
                clearLegalMoves();
            }
//        } else { alert("Invalid move"); }
    }
}

function checkLegalMove(legal_moves, temp_cell) {
  for (let i = 0; i < legal_moves.length; i++) {
    if (legal_moves[i][0] === temp_cell[0] && legal_moves[i][1] === temp_cell[1]) {
      return true;
    }
  }
  return false;
}

function showLegalMoves(legal_moves) {
    const tableBody = document.querySelector('.color-board tbody');
    const rows = tableBody.querySelectorAll('tr');

    if (legal_moves && legal_moves.length > 0) {
        for (let i = 0; i < 8; i++) {
            let cells = rows[i].querySelectorAll('td');
            for (let j = 0; j < 8; j++) {
                temp_cell = [i, j];
                let found = checkLegalMove(legal_moves, temp_cell);
                if (found) {
                    const dot = document.createElement('div');
                    dot.classList.add('dot');
                    cells[j].appendChild(dot);
                }
            }
        }
    }
}

function clearLegalMoves() {
    const dots = document.querySelectorAll('.dot');
    dots.forEach(dot => dot.remove());
}

document.addEventListener("DOMContentLoaded", function() {

  const tableBody = document.querySelector('.color-board tbody');

  function drawBoard() {
    colorBoard.forEach(row => {
      const tr = document.createElement('tr');
      row.forEach(colorIndex => {
        const td = document.createElement('td');
        td.addEventListener('click', handleCellClick);
        td.addEventListener('mouseover', handleCellMouseOver);
        td.addEventListener('mouseout', handleCellMouseOut);
        td.style.backgroundColor = colorDict[colorIndex];
        tr.appendChild(td);
      });
      tableBody.appendChild(tr);
    });
    }

    drawBoard();

    const rows = tableBody.querySelectorAll('tr');

    for (let i = 0; i < 8; i++) {
      const cells = rows[i].querySelectorAll('td');

      for (let j = 0; j < 8; j++) {
            const cellColor = colorBoard[i][j];
            const colorName = colorDict[cellColor];

            if (i == 0) {
                let monk = new Monk("Black", colorName);

                gameBoard[i][j] = monk;
                const circle = createCircle(monk.color, monk.innerColor);
                cells[j].appendChild(circle);

            } else if (i == 7) {
                let monk = new Monk("White", colorName);
                gameBoard[i][j] = monk;
                const circle = createCircle(monk.color, monk.innerColor);
                cells[j].appendChild(circle);

            } else {
                gameBoard[i][j] = null;
            }
      }
    }

    const circles = document.querySelectorAll('.circle');

//    circles.forEach(circle => {
//      circle.addEventListener('click', handleCircleClick);
//    });

//    showLegalMoves(legal_moves);
});

# Sunfish

***

## Installation

* Clone or download the repository
* Navigate into the local repo directory
* Install via ```pip install -e .```

## This Fork uses sunfish in a Gym Environment for AI research

> When I was looking for a chess environment to train AI-Agents on, I couldn't find a finished version, so I decided to create an own one. I found the sunfish chess engine to be perfect for this, as it is quite lightweight and including it in a gym environment seemed straight forward. Currently I use it for selfplay, so I didn't include any 'bot' to play against.

Up to this point, the environment is already usable. It provides the basic stuff so far:

### Action Encoding
This env uses the sunfish way to encode actions. This is a bit fishy (^^) and maybe I will change this:
Each board position has an integer assigned to it:

* **Columns**    
  ```A1, B1, C1, D1 .. --> 91, 92, 93, 94```
  
* **Rows**    
  ```A1, A2, A3, A4 .. --> 91, 81, 71, 61```
  
So **moving** the A2 pawn one step forward (A2 -> A3), would be encoded as the action ```[81, 71]```.

### Rewards

Binary reward: ```+1``` on win, ```-1``` on loose, while the game is in progress, the reward is 0.

### Observation Encoding

An observation is encoded as an 8x8x5 array, where the first plane encodes the piece positions. The board is always rotated to the players perspective. For the player it always looks like he is playing white.
The other four planes ancode the two castling-options for both sides. If castling is possible the according plane contains ones, if not it's all zeros. The first plane in starting position would look like this:

```
[[-4, -2, -3, -5, -6, -3, -2, -4],
 [-1, -1, -1, -1, -1, -1, -1, -1],
 [ 0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0],
 [ 1,  1,  1,  1,  1,  1,  1,  1],
 [ 4,  2,  3,  5,  6,  3,  2,  4]]
 ```
> Note that the player in this case is playing black, so the king (6/-6) and queen (5/-5) positions are inverted.

### Usage

The following methods are implemented:

| Method               | Description                    |
| -------------------- | -------------------------------|
| ```set_opponent```   | Can be used to pass an opponent to the gym to play against. This can be a neural networks ```call``` method. It hast to be a python method that takes in a state and outputs an action. |
| ```reset``` | Resets the board and assigns the player randomly to white or black. If the player is black, the opponent already makes his turn.
| ```possible_moves``` | Returns a list of possible actions for the current player. |
| ```step``` | Takes an action in the form descibed above, applies this action to the board (has to be a valid action), let's the opponent make his turn and returns the resulting ```state```, ```reward``` and ```done```. |
| ```render``` | Renders the current board with the correct coloring of the players, so the black player that has second move is displayed as the black pieces. |

### Credits

* Sunfish is great and was easy to work with, **great python chess engine**.
* The images for the chess pieces are taken from [here](https://commons.wikimedia.org/wiki/File:Chess_Pieces_Sprite.svg) and credit goes to jurgenwesterhof (adapted from work of Cburnett). The images are provided under this [Creative Commons License](https://creativecommons.org/licenses/by-sa/3.0/deed.en).

***

![Sunfish logo](https://raw.github.com/thomasahle/sunfish/master/logo/sunfish_large.png)

## Introduction
Sunfish is a simple, but strong chess engine, written in Python, mostly for teaching purposes. Without tables and its simple interface, it takes up just 111 lines of code!

Because Sunfish is small and strives to be simple, the code provides a great platform for experimenting. People have used it for testing parallel search algorithms, experimenting with evaluation functions, and developing deep learning chess programs. Fork it today and see what you can do!

## Screenshot

    My move: g8f6
    
      8 ♖ ♘ ♗ ♕ ♔ ♗ · ♖
      7 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
      6 · · · · · ♘ · ·
      5 · · · · · · · ·
      4 · · · · ♟ · · ·
      3 · · · · · · · ·
      2 ♟ ♟ ♟ ♟ · ♟ ♟ ♟
      1 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
        a b c d e f g h


    Your move:

# Run it!

Sunfish is self contained in the `sunfish.py` file from the repository. I recommend running it with `pypy` or `pypy3` for optimal performance.

It is also possible to run Sunfish with a graphical interface, such as [PyChess](http://pychess.org), [Arena](http://www.playwitharena.com) or your chess interface of choice. Sunfish' can communicate through the [XBoard](http://www.gnu.org/software/xboard/)/CECP protocol by the command `pypy -u xboard.py`. Ruxy Sylwyka has [a note on making it all work on Windows](http://www.talkchess.com/forum/viewtopic.php?topic_view=threads&p=560462).

[Play now on Lichess!](https://lichess.org/@/sunfish_rs) (requires log in) against [Recursing's Rust port](https://github.com/Recursing/sunfish_rs)

# Features

1. Built around the simple, but deadly efficient MTD-bi search algorithm.
2. Filled with classic as well as modern 'chess engine tricks' for simpler and faster code.
3. Easily adaptive evaluation function through Piece Square Tables.
4. Uses standard Python collections and data structures for clarity and efficiency.

# Limitations

Sunfish supports castling, en passant, and promotion. It doesn't however do minor promotios to rooks, knights or bishops - all input must be done in simple 'two coordinate' notation, as shown in the screenshot.

There are many ways in which you may try to make Sunfish stronger. First you could change from a board representation to a mutable array and add a fast way to enumerate pieces. Then you could implement dedicated capture generation, check detection and check evasions. You could also move everything to bitboards, implement parts of the code in C or experiment with parallel search!

The other way to make Sunfish stronger is to give it more knowledge of chess. The current evaluation function only uses piece square tables - it doesn't even distinguish between midgame and endgame. You can also experiment with more pruning - currently only null move is done - and extensions - currently none are used. Finally Sunfish might benefit from a more advanced move ordering, MVV/LVA and SEE perhaps?

An easy way to get a strong Sunfish is to run with with the [PyPy Just-In-Time intepreter](https://pypy.org/). In particular the python2.7 version of pypy gives a 250 ELO boost compared to the cpython (2 or 3) intepreters at fast time controls:

    Rank Name                    Elo     +/-   Games   Score   Draws
       1 pypy2.7 (7.1)           166      38     300   72.2%   19.7%
       2 pypy3.6 (7.1)            47      35     300   56.7%   21.3%
       3 python3.7               -97      36     300   36.3%   20.7%
       4 python2.7              -109      35     300   34.8%   24.3%


# Why Sunfish?

The name Sunfish actually refers to the [Pygmy Sunfish](http://en.wikipedia.org/wiki/Pygmy_sunfish), which is among the very few fish to start with the letters 'Py'. The use of a fish is in the spirit of great engines such as Stockfish, Zappa and Rybka.

In terms of Heritage, Sunfish borrows much more from [Micro-Max by Geert Muller](http://home.hccnet.nl/h.g.muller/max-src2.html) and [PyChess](http://pychess.org).

# License

[GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.en.html)


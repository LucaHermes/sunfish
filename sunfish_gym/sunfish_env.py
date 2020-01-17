import gym
from gym import spaces, error, utils
from sunfish_gym.sunfish.sunfish import *
import numpy as np
import cv2
import re
import time
import os

my_path = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(my_path, 'images/%s.png')

class SunfishEnv(gym.Env):

	piece_mapping = {
		'p' : -1, 'P' : 1,
		'n' : -2, 'N' : 2,
		'b' : -3, 'B' : 3,
		'r' : -4, 'R' : 4,
		'q' : -5, 'Q' : 5,
		'k' : -6, 'K' : 6,
		'.' : 0
	}

	def __init__(self, opponent=None, play_as_white=None):
		# 8 x 8 observation with a number for each type of piece
		# 2 types of castling for each color
		self.observation_space = spaces.Box(-6, 6, (8, 8, 5))

		#   8 directions * 7 steps = 56 queen-type moves
		# + 8 knight moves = 64 moves
		# for each board location
		self.action_space = spaces.Discrete(((8*7) + 8) * 8 * 8)

		chess_elem = np.concatenate((
		np.ones([1, 80, 80, 4]),
		np.ones([1, 80, 80, 4])*0.4), 1)
		chesscolA = np.concatenate(chess_elem.repeat(4, 0))
		chesscolB = np.roll(np.concatenate(chess_elem.repeat(4, 0)), 80, 0)
		chesscol = np.concatenate((chesscolA[np.newaxis], chesscolB[np.newaxis]), 2)
		self.chessboard_img = np.concatenate(chesscol.repeat(4, 0), 1)
		self.chessboard_img[:,:,3] = 1.

		if opponent is None:
			self.opponent = lambda x,y: self.possible_actions()[np.random.choice(len(self.possible_actions()))]
		else:
			self.opponent = opponent

		self.player_white = play_as_white
		
		# if not given, init random
		if play_as_white is None:
			self.player_white = np.random.randint(2)

		#self.reset()

	def _generate_state(self):
		board = self.board[0].split()
		board = [ [ self.piece_mapping[c] for c in line ] for line in board ]
		board = np.array(board, dtype=np.float32)[:,:,np.newaxis]
		castling_options = np.array([*self.board[2], *self.board[3]], dtype=np.float32)

		#castling_fn = lambda x: x * tf.ones([8, 8, 4], dtype=tf.float32)
		#castling = tf.map_fn(castling_fn, tf.cast(x[:, 1], tf.float32))
		castling_options = castling_options * np.ones([8,8,4], dtype=np.float32)
		return np.concatenate((board, castling_options), -1)

	def set_opponent(self, opponent):
		self.opponent = opponent

	def reset(self):
		self.done = False
		self.player_white = np.random.randint(2)
		# initialize the board, a Position contains:
		# board, score, castling_white, castling_back, en_passant, king_passant
		self.board = Position(initial, 0, (True, True), (True, True), 0, 0)
		self.history = [self.board]
		#self.searcher = Searcher()

		# opponent starts if player is black
		#print(self._generate_state()[:,:,0])
		if not self.player_white:
			valid_ids = self.possible_actions()
			action = self.opponent(self._generate_state(), valid_ids)
			#print(render(action[0]), render(action[1]))
			#print(self.board.value(action))
			#print(action)
			self.sunfish_step(action)

		self.state = self._generate_state()

		return self.state

	def action_to_move(self, action):
		return render(action[0]) + render(action[1])

	def move_to_action(self, move):
		return parse(move[:2]), parse(move[2:])

	def possible_actions(self):
		return np.array(list(self.board.gen_moves()))

	def possible_moves(self):
		possible_actions = self.possible_actions()
		return [ (render(a[0]), render(a[1])) for a in possible_actions ]

	def possible_moves_num(self):
		possible_actions = self.possible_actions()
		possible_moves = np.zeros([*possible_actions.shape, 2])
		for i, a in enumerate(possible_actions):
			for s in [0, 1]:
				rank, fil = divmod(a[s] - A1, 10)
				possible_moves[i][s][0] = -rank
				possible_moves[i][s][1] = fil
		return possible_moves

	def get_piece_mask(self):
		return np.where(self.state[0] != 0, 1, 0)

	def sunfish_step(self, action):
		assert action in self.possible_actions()
		self.board = self.board.move(action)
		#self.board = self.board.rotate()
		return self._generate_state()

	def check_done(self):
		reward = 0.
		done = False

		# if white wins
		if np.abs(self.board.score) >= MATE_LOWER:
			reward = 1.
			done = True
		# if black wins
		#elif self.board.score <= -MATE_LOWER:
		#	reward = 1.
		#	done = True

		#if not self.player_white:
		#	reward *= -1.

		return reward, done


	def step(self, action):
		move = action
		# my color
		self.state = self.sunfish_step(move)
		# their color
		self.reward, self.done = self.check_done()

		if self.done:
			return self.state, self.reward, self.done

		# opponents move
		try:
			valid_ids = self.possible_actions()
			action = self.opponent(self.state, valid_ids)
			self.state = self.sunfish_step(action)
		# their color
		except:
			self.done = True
			print("Draw")

		self.reward, self.done = self.check_done()
		self.reward *= -1
		return self.state, self.reward, self.done


	def render(self, mode='human', close=False):

		im = self.chessboard_img.copy()
		rot = False

		if not self.player_white and not self.reward:
			self.board = self.board.rotate()
			rot = True
		if self.player_white and self.reward:
			self.board = self.board.rotate()
			rot = True
		#if not self.player_white and self.reward not in [0., 1.]:
		#	self.board = self.board.rotate()
		#	rot = True
		#if self.player_white and self.reward == 1.:
		#	self.board = self.board.rotate()
		#	rot = True
		#if self.done and self.reward == 1:
		#	self.board.rotate()
		#	rot = True

		for l, line in enumerate(self.board[0].split()):
			for i, c in enumerate(line):
				if c == '.': continue
				newp = np.zeros([100, 100, 4])
				path = img_path % c
				piece_im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
				h, w = piece_im.shape[:2]
				x, y = (100 - h)//2, (100 - w)//2
				newp[x:x+h, y:y+w] = piece_im / 255.

				piece_im = cv2.resize(newp, (80, 80))
				alpha = piece_im[:,:,-1:].repeat(4, -1)
				frag = im[l*80:(l+1)*80, i*80:(i+1)*80]
				frag = frag*(1-alpha) + piece_im * alpha
				im[l*80:(l+1)*80, i*80:(i+1)*80] = frag

		msg = 'Playing as %s' % ('white' if self.player_white else 'black')
		rew_msg = '         Reward: %.2f' % self.reward
		pos = (10, len(im)-10)
		font = cv2.FONT_HERSHEY_PLAIN
		im = cv2.putText(im, msg + rew_msg, pos, font, 2., (1.,.6,.2), thickness=10)
		im = cv2.putText(im, msg + rew_msg, pos, font, 2., (1.,0.,0.), thickness=4)

		#if self.done and self.reward:

		# rotate back
		if rot: # and self.done:
			self.board = self.board.rotate()

		#im = cv2.putText(im, msg, pos, font, 1., (1.,.6,.2), thickness=6)
		#im = cv2.putText(im, msg, pos, font, 1., (1.,0.,0.), thickness=3)
		cv2.imshow('ChessBoard', im)
		cv2.waitKey(1)
		return im





import matplotlib.pyplot as plt

if __name__ == '__main__':
	env = SunfishEnv(None, False)
	lst_obs = np.copy(env.reset()[0])
	#env.render()
	done = False

	n_games = 0
	n_moves = np.zeros(1000)

	for game in range(1000):
		env.reset()
		done = False
		while not done:
			poss_acts = env.possible_actions()

			if len(poss_acts) == 0:
				break

			action = poss_acts[np.random.choice(len(poss_acts))]
			#print(action)
			#print(env.possible_moves()[:5])
			#print(env.possible_moves_num()[:5])
			#asd
			obs, rew, done = env.step(action)

			#env.render()
			#env.render()
			#time.sleep(1.)

			#lst_obs = np.copy(obs[0])
			n_moves[game] += 1
		n_games += 1

	#cv2.waitKey(0)
	print('Number of games', n_games)
	#ys, xs = np.histogram(n_moves)
	plt.hist(n_moves)
	plt.show()

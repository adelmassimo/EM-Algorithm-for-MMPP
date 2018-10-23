import numpy as np
from numpy import linalg as LA
from scipy.misc import factorial
import model as md

N = 3
Observations = [(4, 1), (3, 1), (5, 2), (2,1), (9, 3), (11, 2), (8, 2), (20, 3), (17, 3), (18, 4)]
D0 = np.matrix('-.4 .4 0; 0 -.5 .5; 0 0 -.6')
D1 = np.eye(N)*.5
M = np.matrix('.7 .2 .1 0; .3 .5 .2 0; .1 .2 .1 .6')

model = md.Model()



def uniformization(Q, q):
	N = np.size(Q[1])
	I = np.eye(N)
	P = I - Q/q
	return P


def randomization( P, q, t ):
	result = np.zeros((N,N))
	#TODO use Fox and Glynn [FG88] to determinae l and r
	l = 0
	r = 500
	for k in range(l, r):
		e = np.exp(-q*t)*np.power(q*t, k)/factorial(k)
		result = result + e * LA.matrix_power(P, k)
	return result


def getAlpha(i):
	N = model.N
	alpha = np.zeros( (1, N) )
	alpha[0] = 1
	
	if i != 0:
		P = randomization(model.P0, model.uniformization_rate, model.observations[i][0])
	
		alpha = np.multiply(
			getAlpha(i-1) * P * model.P1,
		 	np.transpose( model.M[:, model.observations[i][1]] )) 

	return alpha/alpha.sum()


def getBeta(i):
	N = model.N
	M = len(model.observations)
	beta = np.ones( (N, 1) )
	
	if i < M-1:
		P = randomization(model.P0, model.uniformization_rate, model.observations[i+1][0])
		beta = np.multiply(
			P * model.P1 * getBeta(i+1),
			model.M[:, model.observations[i][1]] )
		
	return beta/beta.sum()


def getSubV(i, l):
	return getAlpha(i) * LA.matrix_power( model.P0, l )


def getSubW(i, l):
	return LA.matrix_power( model.P0, l ) * model.P1 * getBeta(i)


def getV(i, r):
	V = np.zeros(( model.N, r-1 ))
	for j in range( r-1 ):
		V[:, j] = getSubV(i, j)
	return V


def getW(i, r):
	W = np.zeros(( r-1, model.N ))
	for j in range( r-1 ):
		W[r-2-j, :] = np.transpose( getSubW(i, j) )
	return W

########################################################################################################################





epsilon = 10
while epsilon > 0.1:
	M = len(model.observations)
	N = model.N
	Z = np.zeros( (N, 4) )
	X0 = np.zeros( (M, N, N) )
	X1 = np.zeros( (M, N, N) )

	for i in range( M - 1 ):
		print i

		X0_tmp = np.zeros( (N, N) )
		X1_tmp = np.zeros( (N, 1) )
		(l, r) = (0, 10)
		a = getAlpha( i )
		b = getBeta( i+1 )
		q = model.uniformization_rate
		t = model.observations[i][0]
		Z[:, model.observations[i][1]] = Z[:, model.observations[i][1]] + a
		
		for k in range(l, r):
			print k
			e = np.exp(-q*t)*np.power(q*t, k)/factorial(k)
			X0_tmp = X0_tmp + e * getV(i, r).dot(getW(i, r)) 
			X1_tmp = X1_tmp + e * np.multiply(getSubV(i, r), np.transpose(b) )

		X0[i,:,:] = X0_tmp.dot( model.P0 ) 
		X1[i,:,:] = np.multiply(X1_tmp, model.P1 )
		# print X1[i,:,:]

	Y0 = X0.sum(axis=0)
	Y1 = X1.sum(axis=0)
	# print Y1

	normFactor = Y0.sum(axis=1) + Y1.sum(axis=1)
	Y0 = Y0 / normFactor[:, np.newaxis]
	Y1 = Y1 / normFactor[:, np.newaxis]
	Z = Z / Z.sum(axis=1)[:, np.newaxis]
	
	print(Y0)
	print(Y1)
	print(Z)
	epsilon = max( LA.norm( model.P0 - Y0, 2),
	    	 	   LA.norm( model.P1 - Y1, 2))
	model.P0 = Y0
	model.P1 = Y1 
	model.M = Z
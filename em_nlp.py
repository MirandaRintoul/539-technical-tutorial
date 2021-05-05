import numpy
from numpy import linalg as la
import numpy.ma as ma
from copy import copy
import nltk
#nltk.download("treebank")

def main():
    # Get data
    tokens, tags, observations = read_nltk_data()
    
    # Initialize parameters
    pi, A, B = initialize_probs(tokens, tags)
    
    # Train model
    new_pi, new_A, new_B = forward_backward(A, B, pi, observations, tokens)



# Read in penn treebank tagged data from nltk package
def read_nltk_data():
    tagged_sentences = nltk.corpus.treebank.tagged_sents()

    # Separate corpus into tokens and tags
    sentence_tokens = []
    sentence_tags = []
    # Train on first 20 for the sake of time
    for s in tagged_sentences[0:20]:
        tokens = []
        tags = []
        for w in s:
            tokens.append(w[0])
            tags.append(w[1])
        sentence_tokens.append(tokens)
        sentence_tags.append(tags)    
        
    # Get all tags and all tokens
    all_tokens = list(set(sum(sentence_tokens, [])))
    all_tags = list(set(sum(sentence_tags, [])))
    
    # Return all tags/tokens, as well as training obs
    return (all_tokens ,all_tags, sentence_tokens)


# Initialize pi, A, and B to be uniform
def initialize_probs(tokens, tags):
    n_tokens = len(tokens)
    n_tags = len(tags)
    
    pi = numpy.ones(n_tags)/n_tags
    A = numpy.ones([n_tags, n_tags])
    A = A / A.sum(axis=1)[:,None]
    B = numpy.ones([n_tags, n_tokens])
    B = B / B.sum(axis=1)[:,None]
    return (pi, A, B)


# Build forward probability matrix
def forward_prob(A, B, pi, obs, tokens):
    # Create matrix to hold probabilities for each state and time
    N = len(A)
    T = len(obs)
    probs = numpy.zeros((N, T))
    # Initialize forward probability values for time 0
    for i in range(N):
        probs[i,0] = pi[i]
    
    # Iteratively calculate all other forward probabilities
    for t in range(T)[1:]:
        # Get word at time t
        word = obs[t] 
        word_loc = tokens.index(word)
        for i in range(N):
            obs_prob = B[i][word_loc]
            transition_probs = [row[i] for row in A]
            prev_forward_probs = probs[:,t-1]
            
            probs[i,t] = obs_prob * sum(prev_forward_probs * transition_probs)
            
    return probs


# Build backward probability matrix
def backward_prob(A, B, pi, obs, tokens):
    # Create matrix to hold probabilities for each state and time
    N = len(A)
    T = len(obs)
    probs = numpy.zeros((N, T))
    # Initialize forward probability values for time 0
    probs[:,T-1] = 1
    
    # Iteratively calculate all other backward probabilities
    for t in reversed(range(T-1)):
        next_word = obs[t+1]
        next_word_loc = tokens.index(next_word)
        for i in range(N):
            transition_probs = [row[i] for row in A]
            observation_probs = [row[next_word_loc] for row in B]
            next_backward_probs = probs[:,t+1]
            
            probs[i, t] = sum(next_backward_probs * transition_probs * observation_probs)
    
    return probs


# Probability of being in state i at time t
def gamma(state, A, B, pi, time, forward_probs, backward_probs):
    # Find probability of full observation sequence
    full_obs = sum(forward_probs[:,time] * backward_probs[:,time])
    
    # Forward + backward probability of given state and time
    forward = forward_probs[state,time]
    backward = backward_probs[state,time]
    
    return (forward * backward) / full_obs

# Probability of being in states i, j at times t, t+1
def xi(states, A, B, pi, time, forward_probs, backward_probs, obs, tokens):
    # Find probability of full observation sequence
    full_obs = sum(forward_probs[:,time] * backward_probs[:,time])
    
    # Probabilities of given states and times
    forward = forward_probs[states[0],time]
    transition = A[states[0]][states[1]]
    backward = backward_probs[states[1],time]
    
    next_word = obs[time+1]
    next_word_loc = tokens.index(next_word)
    observation = B[states[1]][next_word_loc]
    
    return (forward * transition * backward * observation) / full_obs

# Full Baum-Welch (forward-backward) algorithm
def forward_backward(A, B, pi, obs_seqs, tokens):
    # Convergence criteria and max iterations
    eps = .0001
    A_diff = 1
    B_diff = 1
    maxit = 100
    n_iter = 0
    
    V = len(B[0])                           # number of words in vocabulary
    R = len(obs_seqs)                       # number of training sequences
    N = len(A)                              # number of states
    L = max([len(r) for r in obs_seqs])     # length of longest sequence
    
    # Initialize arrays to hold all values of gamma and xi
    gammas = numpy.zeros((R, N, L))
    xis = numpy.zeros((R, N, N, L-1))
    
    # Loop until both A and B have converged, or until max iter is hit
    while ( (A_diff > eps) or (B_diff > eps) ) and (n_iter < maxit):
        # E-step
        for r in range(R):
            obs = obs_seqs[r]  # current observation
            T = len(obs)   # length of current observation
            
            # Get forward/backward probability matrices
            forward = forward_prob(A, B, pi, obs, tokens)
            backward = backward_prob(A, B, pi, obs, tokens)
            
            # Find gamma for each state, xi for each pair of states
            for t in range(T):
                for i in range(N):
                    gammas[r][i][t] = gamma(i, A, B, pi, t, forward, backward)
            for t in range(T-1):
                for i in range(N):
                    for j in range(N):
                        xis[r][i][j][t] = xi([i, j], A, B, pi, t, forward, backward, obs, tokens)
        
        # M-step
        # Store old matrices to compare
        old_A = copy(A)
        old_B = copy(B)
        
        for i in range(N):
            # Update pi
            pi[i] = sum(gammas[:,i,1]) / R
            
            for j in range(N):
                # Update A
                A[i, j] = sum(sum(xis[:,i,j,:])) / sum(sum(gammas[:,i,:-1]))
                
            for v in range(V): 
                current_word = tokens[v]
                obs_total = numpy.zeros((R))
                for r in range(R):
                    # Need to check if obs contains current vocab word
                    mask = [x != current_word for x in obs_seqs[r]]
                    mask += [True] * (L - len(obs_seqs[r]))
                    matches = ma.masked_array(gammas[r,i,:], mask=mask)

                    obs_total[r] = sum(matches)
                    
                # Update B
                B[i, v] = sum(obs_total) / sum(sum(gammas[:,i,:]))
        
        # Find differences in norm of A, B
        A_diff = abs(la.norm(A) - la.norm(old_A))
        B_diff = abs(la.norm(B) - la.norm(old_B))
        n_iter += 1
            
    return (pi, A, B)

if __name__ == "__main__":
    main()




include("policies.jl")

function belief_update(obs,action,b,S,O,A,p_emis,p_trans) 
    """ Compute belief update (3).
    
        Parameters
        ----------
        obs : Int64
            initial observation
    
        action : Int64
            initial action
    
        b : Array{Float64} (length(S))
            previous belief state
    
        S : Array{Int64} 
            range of states
    
        O : Array{Int64}
            range of observations
    
        A : Array{Int64})
            range of actions
        
        p_emis : Array{Float64} (length(S) x length(O))
            emission probability distribution p(o|s)
        
        p_trans : Array{Float64} (length(S) x length(A) x length(S)) 
            transition probability distribution p(s'|s,a)
    
        Returns
        ----------
        b_prime : Array{Float64} (length(S))
            updated belief state
    """
    b_prime = zeros(length(S))
    for s in S
        b_prime[s] = p_emis[action,s,obs]*sum(p_trans[ss,action,s]*b[ss] for ss in S)
    end
    if sum(b_prime) == 0
        b_prime[:] .= 0
    else
        b_prime .= b_prime ./ sum(b_prime)
    end
    return b_prime
end

function simulation(policy,policyPOMDP,Horizon,T,S,O,A,p_init,p_emis,p_trans,reward,tailReward,discount,init_state,init_obs)
    """ Function that simulates a policy.
    
        Parameters
        ----------
        policy : String
            name of the policy belonging to {"SMF", "SARSOP"}
    
        policyPOMDP : policy.alpha
    
        Horizon : Int64
            horizon time of the simulation
    
        T : Int64
            number of optimization time steps
    
        S : Array{Int64}
            range of states
    
        O : Array{Int64}
            range of observations
    
        A : array of actions (Array{Int64}) 
            range of actions
        
        p_init : Array{Float64} (length(S))
            initial state probability distribution p(s)
    
        p_emis : Array{Float64} (length(S) x length(O)
            emission probability distribution p(o|s)
        
        p_trans : Array{Float64} (length(S) x length(A) x length(S)) 
            transition probability distribution p(s'|s,a)
    
        reward : Array{Float64} (length(S) x length(A) x length(S))
            reward function r(s,a,s') 
        
        tailReward : Array{Float64}: (length(S))
            tail reward function v_{MDP}(s)
         
        discount : Float64
            discount factor
    
        Returns
        ----------
        total_reward : Float64 
            accumulated reward at the end of the simulation
        
        policy_time : Float64
            averaged computation time
    """
    # initialize total reward
    total_reward = 0

    # initialize policy computation time
    policy_time = zeros(Horizon)
    
    # initialize belief state
    belief_state = p_init
    b_memory = p_init

    states = ones(Int64,Horizon+1)
    observations = ones(Int64,Horizon+1)
    states[1] = init_state
    observations[1] = init_obs
     
    for t in 1:Horizon

        # save the belief state
        b_memory = belief_state
        ############################################ Optimization block #########################################

        # define the rolling horizon Delta
        Δ = 1
        if Horizon - t <= T
            Δ = Horizon - t +1
        else
            Δ = T
        end

        # choose the action according to a policy (SMF or SARSOP)
        if policy == "SMF"
            action,time_pol = SMF_policy(Δ,S,O,A,belief_state,p_emis,p_trans,reward,tailReward,discount) 
            policy_time[t] = time_pol
            
        elseif policy == "SARSOP"
            action = SARSOP_policy(policyPOMDP,belief_state)
        end
        #################################################################################################
        
        # draw the nex state and next observation given the previous state and the taken action
        states[t+1] = sample(1:nbS,Weights(p_trans[states[t],action,:]))
        observations[t+1] = sample(1:nbO,Weights(p_emis[action,states[t+1],:]))
        total_reward += reward[states[t],action,states[t+1]]*discount^(t-1)
        
        # update the belief state according to (3)
        belief_state = belief_update(observations[t+1],action,b_memory,S,O,A,p_emis,p_trans)
    end    
    return total_reward, mean(policy_time)
end

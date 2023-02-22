using CPUTime
using POMDPs
using POMDPModelTools
using POMDPModels
using POMDPSimulators
using QuickPOMDPs
using POMDPPolicies
using SARSOP

include("models/model_MILP.jl")
include("models/model_tailedMILP.jl")


######################Function that returns action############################
function SMF_policy(T,S,O,A,p_init,p_emis,p_trans,reward,tailReward,discount)
    """ Function that simulates a policy.
    
        Parameters
        ----------
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
        action : Int64 
            action to take
        
        policy_time : Float64
            computation time
    """
    # start the clock
    start = time() 
    
    # define MILP (10) with tail reward function (MDP value function)
    model_milp = model_tailedMILP(T,S,O,A,p_init,p_trans,p_emis,reward,tailReward,discount)

    # solve the MILP
    optimize!(model_milp)

    # define action to take according to SMF policy
    action =1
    obj = 0.0

    if termination_status(model_milp) == MOI.OPTIMAL
        delta = JuMP.value.(model_milp[:delta])

        #Get variable value delta[1,1,a]
        delta1 = zeros(length(A))
        for a in A
            delta1[a] = delta[1,1,a]
            if delta1[a] > 0.5
                action = a
            end
        end

        #Get the objective value
        obj = JuMP.objective_value(model_milp)
    else
        println("The model was not solved correctly. We choose an action randomly.")
        vect = sample(A, 1, replace = false)
        for a in A
            if a in vect
                action = a
            end
        end
    end
    
    # stop the clock
    policy_time = time() - start
    return action, policy_time
end

function def_pomdp(S,O,A,p_init,p_emis,p_trans,reward,discount)
    """ Define a pomdp using the POMDPs.jl framework.
    
        Parameters
        ----------
        S : Array{Int64}
            range of states
    
        O : Array{Int64}
            range of observations
    
        A : array of actions (Array{Int64}) 
            range of actions
        
        p_init : Array{Float64} (length(S))
            initial belief state b(s)
    
        p_emis : Array{Float64} (length(S) x length(O)
            emission probability distribution p(o|s)
        
        p_trans : Array{Float64} (length(S) x length(A) x length(S)) 
            transition probability distribution p(s'|s,a)
    
        reward : Array{Float64} (length(S) x length(A) x length(S))
            reward function r(s,a,s') 
         
        discount : Float64
            discount factor
    
        Returns
        ----------
        pomdp : pomdp class 
            POMDP class defined in POMDPs.jl
    """
    
    # define the function of two variables (because the function of three variables are not taken into account by POMDPs.jl)
    reward_bis = zeros(length(S),length(A))
    for s in S, a in A
        reward_bis[s,a] = sum(reward[s,a,sp]*p_trans[s,a,sp] for sp in S) 
    end

    # define pomdp 
    pomdp = DiscreteExplicitPOMDP(S,A,O,(s,a,sp)->p_trans[s,a,sp],(a,sp,o)->p_emis[a,sp,o],(s,a)->reward_bis[s,a],discount)

    return pomdp
end

function compute_policy(filename,pomdp)
    """ Compute policy for pomdp and write the policy in filename_policy.out.
    
        Parameters
        ----------
        filename : String
            POMDP filename
    
        Returns
        ----------
        pomdp : policy class 
            POMDP policy class defined in POMDPs.jl
    """
    solver = SARSOPSolver(timeout=3600.0,policy_filename=string(filename,"_policy.out"),pomdp_filename=string(filename,".pomdpx"))
    policy = POMDPs.solve(solver,pomdp)
    return policy
end


function SARSOP_policy(policy,p_init)
    """ Compute the action to take according to SARSOP policy.
    
        Parameters
        ----------
        policy : policy class
            POMDP policy
    
        p_init : Array{Float64} (length(S))
            initial belief state b(s)
    
        Returns
        ----------
        a : Int64
            action to take according to SARSOP policy
    """
    a = action(policy,p_init)
    return a
end


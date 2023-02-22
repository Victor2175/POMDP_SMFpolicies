using JuMP
using Gurobi

# define indicator function
function indicator_function(observation,o)
    if o == observation
        return 1
    else
        return 0
    end
end

# relaxation of a model
function relax_model(model)
    for v in all_variables(model)
      
      if is_integer(v)
        unset_integer(v)
      
      # If applicable, also round the lower and upper bounds if they're not integer.
      elseif is_binary(v)
        unset_binary(v)
        if has_lower_bound(v) && and lower_bound(v) > 0
          set_lower_bound(v, 1)
        else
          set_lower_bound(v, 0)
        end
        if has_upper_bound(v) && and upper_bound(v) < 1
          set_upper_bound(v, 0)
        else
          set_upper_bound(v, 1)
        end
        set_upper_bound(v, 1)
      end
    end
    return model
end

# define MILP (10) with tailed rewards
function model_tailedMILP(T,S,O,A,p_init,p_trans,p_emis,reward,tailRewards,discount)
    """Function that returns the MILP
        Returns [JuMP.model]

        T: number of time steps (Int64)
        S: array of states (Array{Int64})
        O: array of observations (Array{Int64}) 
        A: array of actions (Array{Int64}) 
        p_init: initial state probability distribution p(s) (Array{Float64}: length(S)) 
        p_trans: transition probability distribution p(s'|s,a) (Array{Float64}: length(S) x length(A) x length(S)) 
        p_emis: emission probability distribution p(o|s,a) (Array{Float64}: length(A) x length(S) x length(O)) 
        reward: reward function r(s,a,s') (Array{Float64}: length(S) x length(A) x length(S))
        tailRewards: tail reward function v_{MDP}(s) (Array{Float64}: length(S))
        discount: discount factor (Float64)
    """

    # define the model and set the solver
    model = JuMP.direct_model(Gurobi.Optimizer())
    set_optimizer_attribute(model, "TimeLimit", 3600)
    set_optimizer_attribute(model, "MIPGap", 1e-3)
    set_optimizer_attribute(model, "OutputFlag", 0)
    
    @variables model begin
        μ_aso[1:T,A,S,O]>=0
        μ_soa[1:T,S,O,A]>=0
        μ_sa[1:T,S,A]>=0
        μ_as[1:T+1,A,S]>=0
        μ_so[1:T,S,O]>=0
        μ_s[1:T+1,S] >=0
        delta[1:T,O,A], Bin
        cte[A], Bin
    end
    
    # objective function
    @objective(model,Max, sum( μ_sa[t,s,a]*p_trans[s,a,ss]*reward[s,a,ss]*discount^(t-1) for t in 1:T, s in S, a in A, ss in S) + discount^T*sum(μ_s[T+1,ss]*tailRewards[ss] for ss in S))
    
    # Constraint (7b)
    for o in O, a in A
        @constraint(model,delta[1,o,a] == cte[a])
    end
    
    # Constraint (7c)
    for s in S
        @constraint(model,μ_s[1,s] == p_init[s])
    end
    
    # Constraint (7g)
    for t in 1:T, a in A, s in S, o in O 
        @constraint(model, μ_aso[t,a,s,o] == p_emis[a,s,o]*μ_as[t,a,s])
    end
    
    # McCormick inequalities (9)
    for t in 2:T,s in S,o in O, a in A  a  
        @constraint(model, μ_soa[t,s,o,a] <= delta[t,o,a])
        @constraint(model, μ_soa[t,s,o,a] >= μ_so[t,s,o] - (1 - delta[t,o,a]))
        @constraint(model, μ_soa[t,s,o,a] <= μ_so[t,s,o])
    end

    # Constraint (7h)
    for t in 1:T, o in O
        @constraint(model,sum(delta[t,o,aa] for aa in A) == 1)
    end

    ####################Consistency constraints##########################
    
    # Constraint (7d)
    for t in 2:T+1, s in S
        @constraint(model, sum(μ_as[t-1,aa,s] for aa in A) ==μ_s[t,s])
    end
    
    # Constraint (7f)
    for t in 1:T, a in A, s in S
        @constraint(model, sum(μ_sa[t,ss,a]*p_trans[ss,a,s] for ss in S) ==μ_as[t,a,s])
    end
    
    # Constraint (7e)
    for t in 2:T, s in S, o in O
        @constraint(model, sum(μ_aso[t-1,aa,s,o] for aa in A) ==μ_so[t,s,o])
    end
    
    # Constraint (7e)
    for t in 1:T, s in S, o in O
        @constraint(model, sum(μ_soa[t,s,o,aa] for aa in A) ==μ_so[t,s,o])
    end
    
    # Constraint (7b)
    for s in S, a in A
        @constraint(model,μ_sa[1,s,a] == p_init[s]*cte[a])
    end

    # Constraint (7e)
    for t in 1:T, s in S, a in A
        @constraint(model, sum(μ_soa[t,s,oo,a] for oo in O) ==μ_sa[t,s,a])
    end
    return model
end

# define MILP (10) with tailed rewards (MDP value function) and valid cuts (11)
function model_tailedMILP_cuts(T,S,O,A,p_init,p_trans,p_emis,p_cuts,reward,tailRewards,discount)
    """Function that returns the MILP he valid cuts
        Returns [JuMP.model]

        T: number of time steps (Int64)
        S: array of states (Array{Int64})
        O: array of observations (Array{Int64}) 
        A: array of actions (Array{Int64}) 
        p_init: initial state probability distribution p(s) (Array{Float64}: length(S) x 1) 
        p_trans: transition probability distribution p(s'|s,a) (Array{Float64}: length(S) x length(A) x length(S)) 
        p_emis: emission probability distribution p(o|s) (Array{Float64}: length(S) x length(O))
        p_cuts: independence probabilities p(s'|s,a,o) (Array{Float64}: length(S) x length(A) x length(O) x length(S))  
        reward: reward function r(s,a,s') (Array{Float64}: length(S) x length(A) x length(S))
        tailRewards: tail reward function v_{MDP}(s) (Array{Float64}: length(S))
        discount: discount factor (Float64)
    """

    # define tailed MILP
    model_cuts = model_tailedMILP(T,S,O,A,p_init,p_trans,p_emis,reward,tailRewards,discount)

    @variables model_cuts begin
        μ_sasoa[2:T,S,A,S,O,A] >=0
    end
    
    ########################################### Consistency constraints ############################
    
    # Constraint (11a)
    for t in 2:T, s in S, a in A
        @constraint(model_cuts,sum(μ_sasoa[t,ss,aa,s,oo,a] for ss in S, aa in A, oo in O) ==model_cuts[:μ_sa][t,s,a])
    end
    
    
    # Constraint (11b)
    for t in 2:T, s in S, a in A, ss in S, o in O
        @constraint(model_cuts,sum(μ_sasoa[t,s,a,ss,o,aa] for aa in A) == p_emis[a,ss,o]*p_trans[s,a,ss]*model_cuts[:μ_sa][t-1,s,a])
    end
    ######################################################################################################

    ########################################### Valid cuts ############################
    # Constraint (11c)
    for t in 2:T, s in S, a in A, oo in O, ss in S, aa in A
        @constraint(model_cuts,μ_sasoa[t,s,a,ss,oo,aa] == p_cuts[s,a,ss,oo]*sum(μ_sasoa[t,s,a,sss,oo,aa] for sss in S))
    end
    ####################################################################################################

    return model_cuts
end

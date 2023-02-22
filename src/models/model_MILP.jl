using JuMP
using Gurobi

function relax_model(model)
    """Function that returns the linear relaxation of a model
        Returns [JuMP.model]

        model: the initial JuMP.model
    """
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

function model_MILP(T,S,O,A,p_init,p_trans,p_emis,reward)
    """ Function that returns MILP (10) with finite horizon.
    
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
            initial state belief state b(s)
    
        p_emis : Array{Float64} (length(S) x length(O)
            emission probability distribution p(o|s)
        
        p_trans : Array{Float64} (length(S) x length(A) x length(S)) 
            transition probability distribution p(s'|s,a)
    
        reward : Array{Float64} (length(S) x length(A) x length(S))
            reward function r(s,a,s') 
    
        Returns
        ----------
        model : JuMP.model
            MILP model 
    """

    #Define the model and set the solver
    model = JuMP.direct_model(Gurobi.Optimizer())
    set_optimizer_attribute(model, "TimeLimit", 3600)
    set_optimizer_attribute(model, "MIPGap", 1e-4)
    set_optimizer_attribute(model, "OutputFlag", 0)
    
    @variables model begin
        μ_aso[1:T,A,S,O]>=0
        μ_soa[1:T,S,O,A]>=0
        μ_sas[1:T,S,A,S]>=0
        μ_sa[1:T,S,A]>=0
        μ_as[1:T+1,A,S]>=0
        μ_so[1:T,S,O]>=0
        delta[1:T,O,A], Bin
        cte[A], Bin
    end
    @objective(model,Max, sum(μ_sas[t,s,a,ss]*reward[s,a,ss] for t in 1:T, s in S, a in A, ss in S))
    
    for s in S
        @constraint(model,sum(μ_as[1,aa,s] for aa in A) == p_init[s])
    end

    for t in 1:T, a in A, s in S, o in O
        @constraint(model, μ_aso[t,a,s,o] == p_emis[a,s,o]*μ_as[t,a,s])
    end

    for t in 1:T, s in S,a in A, ss in S
        @constraint(model, μ_sas[t,s,a,ss] == p_trans[s,a,ss]*μ_sa[t,s,a])
    end

    # ## McCormick inequalities
    for t in 1:T,s in S,o in O, a in A
        @constraint(model, μ_soa[t,s,o,a] <= delta[t,o,a])
        @constraint(model, μ_soa[t,s,o,a] >= μ_so[t,s,o] - (1 - delta[t,o,a]))
        @constraint(model, μ_soa[t,s,o,a] <= μ_so[t,s,o])
    end
    
    for t in 1:T, o in O
         @constraint(model,sum(delta[t,o,aa] for aa in A) == 1)
    end

    ########################### Optional constraint ######################
    for o in O, a in A
         @constraint(model,delta[1,o,a] == cte[a])
    end

    for s in S, a in A
         @constraint(model,μ_sa[1,s,a] == p_init[s]*cte[a])
    end
    
    #####################Consistency constraints###########################

    for t in 1:T, a in A, s in S
        @constraint(model, sum(μ_sas[t,ss,a,s] for ss in S) ==μ_as[t+1,a,s])
    end

    for t in 1:T, s in S, o in O
        @constraint(model, sum(μ_aso[t,aa,s,o] for aa in A) ==μ_so[t,s,o])
    end

    for t in 1:T, s in S, o in O
        @constraint(model, sum(μ_soa[t,s,o,aa] for aa in A) ==μ_so[t,s,o])
    end

    for t in 1:T, s in S, a in A
        @constraint(model, sum(μ_soa[t,s,oo,a] for oo in O) ==μ_sa[t,s,a])
    end
    return model
end


# define MILP (10) with valid cuts (11) and finite horizon
function model_MILP_cuts(T,S,O,A,p_init,p_trans,p_emis,p_cuts,reward)
    """ Function that returns MILP (10) with finite horizon and valid cuts (11).
    
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
            initial state belief state b(s)
    
        p_emis : Array{Float64} (length(S) x length(O)
            emission probability distribution p(o|s)
        
        p_trans : Array{Float64} (length(S) x length(A) x length(S)) 
            transition probability distribution p(s'|s,a)
    
        p_cuts : Array{Float64} (length(S) x length(A) x length(S) x length(O)) 
            probability distribution p(s|s',a',o) defined for (11)
    
        reward : Array{Float64} (length(S) x length(A) x length(S))
            reward function r(s,a,s') 
    
        Returns
        ----------
        model : JuMP.model
            MILP model (10) with cuts (11)
    """

    # define MILP (10)
    model_cuts = model_MILP(T,S,O,A,p_init,p_trans,p_emis,reward)

    @variables model_cuts begin
        μ_sasoa[2:T,S,A,S,O,A] >=0
    end
    
    ###########################################Local consistency constraints############################
    for t in 2:T, s in S, a in A, ss in S, o in O
        @constraint(model_cuts,sum(μ_sasoa[t,s,a,ss,o,aa] for aa in A) == p_emis[a,ss,o]*p_trans[s,a,ss]*model_cuts[:μ_sa][t-1,s,a])
    end

    for t in 2:T, s in S, a in A
        @constraint(model_cuts,sum(μ_sasoa[t,ss,aa,s,oo,a] for ss in S, aa in A, oo in O) ==model_cuts[:μ_sa][t,s,a])
    end
    ######################################################################################################

    ###########################################Valid cuts############################
    #μ_sasoa^t = p_(s' | sao)^t * sum(μ_sasoa^{t}, s_t) 
    for t in 2:T, s in S, a in A, oo in O, ss in S, aa in A
        @constraint(model_cuts,μ_sasoa[t,s,a,ss,oo,aa] == p_cuts[s,a,ss,oo]*sum(μ_sasoa[t,s,a,sss,oo,aa] for sss in S))
    end
    ####################################################################################################

    return model_cuts
end

# define NLP model (7) with finite horizon
function model_nlp_gurobi(T,S,O,A,p_init,p_trans,p_emis,reward)
    """ Function that returns NLP (7) with finite horizon T.
    
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
            initial state belief state b(s)
    
        p_emis : Array{Float64} (length(S) x length(O)
            emission probability distribution p(o|s)
        
        p_trans : Array{Float64} (length(S) x length(A) x length(S)) 
            transition probability distribution p(s'|s,a)
    
        p_cuts : Array{Float64} (length(S) x length(A) x length(S) x length(O)) 
            probability distribution p(s|s',a',o) defined for (11)
    
        reward : Array{Float64} (length(S) x length(A) x length(S))
            reward function r(s,a,s') 
    
        Returns
        ----------
        model : JuMP.model
            NLP model (7) 
    """

    #Define the model and set the solver
    model = JuMP.direct_model(Gurobi.Optimizer())
    set_optimizer_attribute(model, "TimeLimit", 3600)
    set_optimizer_attribute(model, "MIPGap", 1e-4)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "NonConvex", 2)
    

    @variables model begin
        μ_aso[1:T,A,S,O]>=0
        μ_soa[1:T,S,O,A]>=0
        μ_sas[1:T,S,A,S]>=0
        μ_sa[1:T,S,A]>=0
        μ_as[1:T+1,A,S]>=0
        μ_so[1:T,S,O]>=0
        delta[1:T,O,A], Bin
        cte[A], Bin
    end
    @objective(model,Max, sum( μ_sas[t,s,a,ss]*reward[s,a,ss] for t in 1:T, s in S, a in A, ss in S))

    for s in S
        @constraint(model,sum(μ_as[1,aa,s] for aa in A) == p_init[s])
    end

    for t in 1:T, a in A, s in S, o in O
        @constraint(model, μ_aso[t,a,s,o] == p_emis[a,s,o]*μ_as[t,a,s])
    end

    for t in 1:T, s in S,a in A, ss in S
        @constraint(model, μ_sas[t,s,a,ss] == p_trans[s,a,ss]*μ_sa[t,s,a])
    end

    # nonlinear constraints
    for t in 1:T,s in S,o in O, a in A
        @constraint(model, μ_soa[t,s,o,a] == μ_so[t,s,o]*delta[t,o,a])
    end

    for t in 1:T, o in O
         @constraint(model,sum(delta[t,o,aa] for aa in A) == 1)
    end

    ########################### Optional constraint ######################
    for o in O, a in A
         @constraint(model,delta[1,o,a] == cte[a])
    end

    #####################Consistency constraints###########################

    for t in 1:T, a in A, s in S
        @constraint(model, sum(μ_sas[t,ss,a,s] for ss in S) ==μ_as[t+1,a,s])
    end

    for t in 1:T, s in S, o in O
        @constraint(model, sum(μ_aso[t,aa,s,o] for aa in A) ==μ_so[t,s,o])
    end

    for t in 1:T, s in S, o in O
        @constraint(model, sum(μ_soa[t,s,o,aa] for aa in A) ==μ_so[t,s,o])
    end

    for t in 1:T, s in S, a in A
        @constraint(model, sum(μ_soa[t,s,oo,a] for oo in O) ==μ_sa[t,s,a])
    end
    return model
end

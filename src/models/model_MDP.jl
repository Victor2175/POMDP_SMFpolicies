import JuMP
import Gurobi

function model_MDP(S,A,p_init,p_trans,reward,discount)
    """Function that returns MDP linear program.
        Returns [JuMP.model]

        S: array of states (Array{Int64})
        A: array of actions (Array{Int64})
        p_init: initial state probability distribution p(s) (Array{Float64}:length(S))
        p_trans: transition probability distribution p(s'|s,a) (Array{Float64}: length(S) x length(A) x length(S)) 
        reward: reward function r(s,a,s') (Array{Float64}: length(S) x length(A) x length(S))
    """

    #Define the model and set the solver
    model = JuMP.direct_model(Gurobi.Optimizer())
    set_optimizer_attribute(model, "TimeLimit", 3600)
    set_optimizer_attribute(model, "MIPGap", 1e-4)
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variables model begin
        v_s[S]>=0
    end
    @objective(model,Min, sum(v_s[s]*p_init[s] for s in S))


    for s in S, a in A
        @constraint(model,v_s[s] >= sum(reward[s,a,ss]*p_trans[s,a,ss] + discount*v_s[ss]*p_trans[s,a,ss] for ss in S))
    end

    return model
end

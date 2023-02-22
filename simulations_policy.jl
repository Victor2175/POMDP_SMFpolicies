using Printf
using Distributed
using Statistics
using CPUTime
using StatsBase
include("parser.jl")
include("simulations.jl")
include("model_simPOMDP.jl")
include("model_MDP.jl")

using Random

Random.seed!(1234)

filename = ARGS[1]
nb_sims = parse(Int,ARGS[2])
Horizon = parse(Int,ARGS[3])
T = parse(Int,ARGS[4])
T_bound = parse(Int, ARGS[5])


################Read the instance and load it##############################################
S, O, A, p_init,p_trans, p_emis, reward = parser_POMDP(string("instances/",filename,".dat"))

# compute the initial belief state
p_init = p_init./sum(p_init)
p_trans = p_trans./sum(p_trans,dims=3)
p_emis = p_emis./sum(p_emis,dims=3)

discount=0.95


nbS = length(S)
nbO = length(O)
nbA = length(A)


##########################Compute the probabilities for independence cuts###########
# #p(s_{t}|s_{t-1}a_{t-1}o_t)
p_cuts = zeros(length(S),length(A),length(S), length(O))

for s in S, a in A, ss in S, oo in O
    if sum(p_trans[s,a,sss]*p_emis[a,sss,oo] for sss in S) == 0
	if ss == 1
                p_cuts[s,a,ss,oo] = 1.0
        else
                p_cuts[s,a,ss,oo] = 0.0
        end
    else
        p_cuts[s,a,ss,oo] = p_trans[s,a,ss]*p_emis[a,ss,oo]/sum(p_trans[s,a,sss]*p_emis[a,sss,oo] for sss in S)
    end
end

#println(tailRewards)
println(sum(p_trans,dims=3))

############################### Compute MDP value function with value #####
tailRewards = zeros(nbS)
mdp = model_MDP(S,A,p_init,p_trans,reward,discount)
optimize!(mdp)
v_function = JuMP.value.(mdp[:v_s])
for s in S
	tailRewards[s] = v_function[s]
end

################Function to launch simulations (in parallel)###############
############################################################################
############################################################################

init_state = zeros(Int64, nb_sims)
init_obs = zeros(Int64, nb_sims)
belief_state = ones(Float64,nb_sims,nbS)

p_uniform = (1/nbS)* ones(nbS)

for sim in 1:nb_sims
	init_state[sim] = sample(S,Weights(p_uniform))
	init_obs[sim] = sample(O,Weights(p_emis[1,init_state[sim],:]))
	belief_state[sim,:] = p_init
end


###################### Compute upper bounds Large horizon ######################################
model_lp = model_sim_tailed(T_bound,S,O,A,p_init,p_trans,p_emis,reward,tailRewards,discount,1)
model_lp = relax_model(model_lp)
optimize!(model_lp)
obj_lp = objective_value(model_lp)
opt_status = termination_status(model_lp)
println(opt_status)
obj_bound_lp = JuMP.objective_bound(model_lp)
println(obj_bound_lp)
println(JuMP.has_duals(model_lp))
println(JuMP.dual_status(model_lp))
#println(JuMP.dual_objective_value(model_lp))
#println(obj_dual_lp)

model_lp_cuts = model_sim_tailed_cuts(T_bound,S,O,A,p_init,p_trans,p_emis,p_cuts,reward,tailRewards,discount,1)
model_lp_cuts = relax_model(model_lp_cuts)
optimize!(model_lp_cuts)
obj_lp_cuts = objective_value(model_lp_cuts)
opt_cuts_status = termination_status(model_lp_cuts)
println(opt_cuts_status)

if opt_cuts_status== "TIME_LIMIT"
	obj_lp_cuts = JuMP.objective_bound(model_lp_cuts)
end

#obj_dual_lp_cuts = JuMP.dual_objective_value(model_lp_cuts)
#println(obj_dual_lp_cuts)


###################### Compute upper bounds small horizon #########################
tt_bd = 20 
model_lp = model_sim_tailed(tt_bd,S,O,A,p_init,p_trans,p_emis,reward,tailRewards,discount,1)
model_lp = relax_model(model_lp)
optimize!(model_lp)
obj_lp_smallT = objective_value(model_lp)
opt_status = termination_status(model_lp)
println(opt_status)
#obj_bound_lp = JuMP.objective_bound(model_lp)

model_lp_cuts = model_sim_tailed_cuts(tt_bd,S,O,A,p_init,p_trans,p_emis,p_cuts,reward,tailRewards,0.95,1)
model_lp_cuts = relax_model(model_lp_cuts)
optimize!(model_lp_cuts)
obj_lp_cuts_smallT = objective_value(model_lp_cuts)
opt_cuts_status = termination_status(model_lp_cuts)
println(opt_cuts_status)

if opt_cuts_status== "TIME_LIMIT"
        obj_lp_cuts_smallT = JuMP.objective_bound(model_lp_cuts)
end

#obj_bound_lp_cuts = JuMP.objective_bound(model_lp_cuts)


#######################################################################################################
@printf "=============== POMDP policies on discounted infinite horizon =======\n"
pomdp = def_pomdp(S,O,A,p_init,p_emis,p_trans,reward,discount)
start = time()
SARSOPpolicy = compute_policy(filename,pomdp,"SARSOP")
SARSOP_time = time()-start
#######################################################################################################

# Rewards of SMF policy
SMF_reward = zeros(nb_sims)
SMF_time = zeros(nb_sims)

# Reward of SARSOP policy
SARSOP_reward = zeros(nb_sims)
SARSOP_time = SARSOP_time/nb_sims

#Rewards of LP policy
LP_reward = zeros(nb_sims)
LP_time = zeros(nb_sims)

#Rewards of LP policy with cuts
reward_LPcuts = zeros(nb_sims)
policy_times_LPcuts = zeros(nb_sims)

for sim in 1:nb_sims	
	@printf "================Start simulation %.0f ============================\n" sim 
	
	###########################################################
	
	Random.seed!(sim)
	@printf "=============Run SMF policy T=%.0f ================== \n" T
	######## Run SMF policy ########
	policy_reward, policy_time = simulation("MILP_tailed",SARSOPpolicy,Horizon,T,S,O,A,p_init,p_trans,p_emis,p_cuts,reward,tailRewards,discount,init_state[sim],init_obs[sim])
	SMF_reward[sim] = policy_reward
	SMF_time[sim] = policy_time
	#####################################################

	Random.seed!(sim)
        @printf "=============Run SARSOP policy ================== \n" 
        ########Run MILP heuristic with cuts########
        policy_reward, policy_time = simulation("SARSOP",SARSOPpolicy,Horizon,T,S,O,A,p_init,p_trans,p_emis,p_cuts,reward,tailRewards,discount,init_state[sim],init_obs[sim])
        SARSOP_reward[sim] = policy_reward
        #####################################################

	@printf "=============Run SMF policy with relaxation ================== \n"
	Random.seed!(sim)
	########Run LP policy with cuts########
        #policy_reward, policy_time = simulation("LP_tailed_cuts",SARSOPpolicy,Horizon,T,S,O,A,p_init,p_trans,p_emis,p_cuts,reward,tailRewards,discount,init_state[sim],init_obs[sim])
        LP_reward[sim] = policy_reward
        LP_time[sim] = policy_time

	
	if sim%100==0

		best_bound = [min(obj_lp_cuts,obj_lp) for i in 1:nb_sims]
		best_bound_smallT = [min(obj_lp_cuts_smallT,obj_lp_smallT) for i in 1:nb_sims]

		# compute gaps on smaller horizon (larger gaps)
		SMF_gap_smallT = 100*(best_bound_smallT - SMF_reward)./abs.(best_bound_smallT)
		SARSOP_gap_smallT = 100*(best_bound_smallT - SARSOP_reward)./abs.(best_bound_smallT)
		LP_gap_smallT = 100*(best_bound_smallT - LP_reward)./abs.(best_bound_smallT)

		# compute gaps on larger horizon (smaller gaps)
		SMF_gap = 100*(best_bound - SMF_reward)./abs.(best_bound)
		SARSOP_gap = 100*(best_bound - SARSOP_reward)./abs.(best_bound)
		LP_gap = 100*(best_bound - LP_reward)./abs.(best_bound)



		createFolder(string("simulations_",filename))
		outfile = string("simulations_",filename,"/simulations_",filename,"_",T,"_",T_bound,".dat")

		f = open(outfile, "w")
		println(f,string(" Upper bound with cuts T=20 : ", obj_lp_cuts_smallT))
		println(f,string(" Upper bound T=20 : ", obj_lp_smallT))
		println(f,"\n")
		println(f,string(" Upper bound with cuts T=100 : ", obj_lp_cuts))
		println(f,string(" Upper bound T=100 : ", obj_lp))
		println(f,"\n")
		println(f,string("===== SMF policy =====\n"))
		println(f,string("SMF policy reward : ",mean(SMF_reward)))
		println(f,string("SMF policy gap T=20 : ",mean(SMF_gap_smallT)))
		println(f,string("SMF policy gap T=100 : ",mean(SMF_gap)))
		println(f,string("SMF policy time : ",mean(SMF_time)))
		println(f,"\n")
		println(f,string("===== SARSOP policy =====\n"))
		println(f,string("SARSOP policy reward : ",mean(SARSOP_reward)))
		println(f,string("SARSOP policy gap T=20 : ",mean(SARSOP_gap_smallT)))
		println(f,string("SARSOP policy gap T=100 : ",mean(SARSOP_gap)))
		println(f,string("SARSOP policy time : ",SARSOP_time))
		println(f,"\n")
	
		close(f)
	end
end

best_bound = [min(obj_lp_cuts,obj_lp) for i in 1:nb_sims]
best_bound_smallT = [min(obj_lp_cuts_smallT,obj_lp_smallT) for i in 1:nb_sims]

# compute gaps on smaller horizon (larger gaps)
SMF_gap_smallT = 100*(best_bound_smallT - SMF_reward)./abs.(best_bound_smallT)
SARSOP_gap_smallT = 100*(best_bound_smallT - SARSOP_reward)./abs.(best_bound_smallT)
LP_gap_smallT = 100*(best_bound_smallT - LP_reward)./abs.(best_bound_smallT)

# compute gaps on larger horizon (smaller gaps)
SMF_gap = 100*(best_bound - SMF_reward)./abs.(best_bound)
SARSOP_gap = 100*(best_bound - SARSOP_reward)./abs.(best_bound)
LP_gap = 100*(best_bound - LP_reward)./abs.(best_bound)


@printf "============================================================================\n"
@printf "===================================================================\n"
@printf "===========================================================\n"
@printf "==============================================\n"
@printf "============= Results ==================\n"

@printf "Upper bound with cuts T=20 : %f \n" obj_lp_cuts_smallT
@printf "Upper bound T=20 : %f \n" obj_lp_smallT

@printf "Upper bound with cuts T=100 : %f \n" obj_lp_cuts
@printf "Upper bound T=100 : %f \n" obj_lp

@printf "===== SMF policy T=%.0f =====\n" T
@printf "SMF policy gap T=20 : %f \n" mean(SMF_gap_smallT)
@printf "SMF policy gap T=100 : %f \n" mean(SMF_gap)
@printf "SMF policy reward : %f \n" mean(SMF_reward)
@printf "SMF policy time : %f \n\n" mean(SMF_time)

@printf "===== LP policy T=%.0f =====\n" T
@printf "LP policy gap T=20 : %f \n" mean(LP_gap_smallT)
@printf "LP policy gap T=100 : %f \n" mean(LP_gap)
@printf "LP policy reward : %f \n" mean(LP_reward)
@printf "LP policy time : %f \n\n" mean(LP_time)

@printf "===== SARSOP policy T=%.0f =====\n" T
@printf "SARSOP policy gap T=20 : %f \n" mean(SARSOP_gap_smallT)
@printf "SARSOP policy gap T=100 : %f \n" mean(SARSOP_gap)
@printf "SARSOP policy reward : %f \n" mean(SARSOP_reward)
@printf "SARSOP policy time : %f \n\n" SARSOP_time


createFolder(string("simulations_",filename))
outfile = string("simulations_",filename,"/simulations_",filename,"_",T,"_",T_bound,".dat")

f = open(outfile, "w")
println(f,string(" Upper bound with cuts T=20 : ", obj_lp_cuts_smallT))
println(f,string(" Upper bound T=20 : ", obj_lp_smallT))
println(f,"\n")
println(f,string(" Upper bound with cuts T=100 : ", obj_lp_cuts))
println(f,string(" Upper bound T=100 : ", obj_lp))
println(f,"\n")
println(f,string("===== SMF policy =====\n"))
println(f,string("SMF policy reward : ",mean(SMF_reward)," (+/- ",std(SMF_reward) ,")"))
println(f,string("SMF policy gap T=20 : ",mean(SMF_gap_smallT)," (+/- ",std(SMF_gap_smallT) ,")"))
println(f,string("SMF policy gap T=100 : ",mean(SMF_gap)," (+/- ",std(SMF_gap) ,")"))
println(f,string("SMF policy time : ",mean(SMF_time)," (+/- ",std(SMF_time) ,")"))
println(f,"\n")
println(f,string("===== SARSOP policy =====\n"))
println(f,string("SARSOP policy reward : ",mean(SARSOP_reward)," (+/- ",std(SARSOP_reward) ,")"))
println(f,string("SARSOP policy gap T=20 : ",mean(SARSOP_gap_smallT)," (+/- ",std(SARSOP_gap_smallT) ,")"))
println(f,string("SARSOP policy gap T=100 : ",mean(SARSOP_gap)," (+/- ",std(SARSOP_gap) ,")"))
println(f,string("SARSOP policy time : ",SARSOP_time))
println(f,"\n")

close(f)

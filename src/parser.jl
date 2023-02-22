function createFolder(directory)
    """ Create a folder in a directory if it doesn't exist yet.
    
        Parameters
        ----------
        directory : String
            directory name
    """
    try
        if isdir(directory) == false
            mkdir(directory)
        end
    catch
        println(string("Error:Creating directory ", directory))
    end
end

function parser_POMDP(filename)
    """ Parse the file "filename.dat" and load the state space S, action space A, observation space O, and the probability distributions p_init, p_emis, p_trans the reward function reward.
        
        Parameters
        ----------
        filename : String
            POMDP filename
    """
    f = open(string(filename))
    nb_line = 1
    curr_matrix = "!"
    a = 1
    s = 1
    nbS = 1
    nbA = 1
    nbO = 1
    p_init = zeros(nbS)
    p_emis = zeros(nbS, nbO)
    p_trans = zeros(nbS, nbA, nbS)
    reward = zeros(nbS, nbA, nbS)
        
    # read the file    
    for ln in eachline(f)
        txt_data = split(ln)

        if nb_line ==1
            nbS = parse(Int, txt_data[3])
        end
        if nb_line ==2
            nbO = parse(Int, txt_data[3])
        end
        if nb_line ==3

            nbA = parse(Int, txt_data[3])
            p_init = zeros(nbS)
            p_emis = zeros(nbA,nbS, nbO)
            p_trans = zeros(nbS, nbA, nbS)
            reward = zeros(nbS, nbA, nbS)
        end

        if isempty(txt_data) == true
            continue
        end

        if txt_data[1] == "P_init"
            curr_matrix = "P_init"
            continue
        end

        if curr_matrix == "P_init"

            for ss in 1:nbS
                p_init[ss] =  parse(Float64,txt_data[ss])
            end
            curr_matrix = "P_trans"
            continue
        end
        
        if txt_data[1] == "P_trans"
            continue
        end

        if curr_matrix == "P_trans"

            for ss in 1:nbS
                p_trans[s,a,ss] =  parse(Float64,txt_data[ss])
            end
            a = a+1
            if a == nbA +1
                a = 1
                s = s+1
            end

            if s == nbS +1
                s = 1
                curr_matrix = "P_emis"
                continue
            end
        end

        if txt_data[1] == "P_emis"
            continue
        end
        if curr_matrix == "P_emis"

            for o in 1:nbO
                p_emis[a,s,o] =  parse(Float64,txt_data[o])
            end
            s = s+1
            if s == nbS +1
                s = 1
                a = a+1
            end
            if a == nbA +1
                a = 1
                curr_matrix = "Reward"
                continue
            end
        end

        if txt_data[1] == "Reward"
            continue
        end

        if curr_matrix == "Reward"

            for ss in 1:nbS
                reward[s,a,ss] =  parse(Float64,txt_data[ss])
            end
            a = a+1
            if a == nbA +1
                a = 1
                s = s+1
            end

            if s == nbS +1
                s = 1
                curr_matrix = "!"
            end
        end
        nb_line += 1
    end
    close(f)
    S = 1:nbS
    A = 1:nbA
    O = 1:nbO
    return S, O, A, p_init, p_emis, p_trans, reward
end

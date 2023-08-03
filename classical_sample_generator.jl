#=
test:
- Julia version: 1.8.0
- Author: andy
- Date: 2022-09-02
=#
using LevyArea
using DelimitedFiles

function gen_samples(;its::Int64 = 65536,w_dim::Int64 = 2,h:: Float64 = 1.0,
    err:: Float64 = 0.0001,fixed:: Bool = false,
    W:: Array{Float64} = [1.0,-0.5,-1.2,-0.3,0.7,0.2,-0.9,0.1,1.7],
    num_chen_combinations:: Int64 = 0,
    filename = "")

    resDim = Int64(w_dim*(w_dim+1)/2)
    results = Array{Float64}(undef,its,resDim)
    W = W[1:w_dim]
    multiplier = sqrt(2)^num_chen_combinations
    W *= multiplier

    for i in 1:its

        if fixed == false
           W = randn(w_dim)
        end
        II = iterated_integrals(W,h,err)

        idx:: Int = w_dim+1
        for k in 1:w_dim
            results[i,k] = W[k]
            #println("wiritng results[$i,$k] = $(W[k])")
            for l in (k+1):w_dim
                a = 0.5*(II[k,l] - II[l,k])
                results[i,idx] = a
                idx +=1
            end
        end

        # if i%100 == 0
        #     println(i)
        # end
    end

    if filename == ""
        filename = "samples/samples_$w_dim-dim.csv"
        if fixed
            filename = "samples/fixed_samples_$w_dim-dim.csv"
        end
    end

    # filename = "high_prec_samples.csv"

    writedlm(filename, results, ',')
end

function gen_true_levy(;its::Int64 = 65536,w_dim::Int64 = 2,
    p:: Int = 3,fixed:: Bool = false,
    W:: Array{Float64} = [1.0,-0.5,-1.2,-0.3,0.7,0.2,-0.9,0.1,1.7],
    filename = "")

    resDim = Int64(w_dim*(w_dim+1)/2)
    results = Array{Float64}(undef,its,resDim)
    W = W[1:w_dim]

    for i in 1:its

        if fixed == false
           W = randn(w_dim)
        end
        II = LevyArea.levyarea(W, p, LevyArea.Milstein())

        idx:: Int = w_dim+1
        for k in 1:w_dim
            results[i,k] = W[k]
            #println("wiritng results[$i,$k] = $(W[k])")
            for l in (k+1):w_dim
                a = II[k,l]
                results[i,idx] = a
                idx +=1
            end
        end

        # if i%100 == 0
        #     println(i)
        # end
    end

    if filename == ""
        filename = "samples/samples_$w_dim-dim.csv"
        if fixed
            filename = "samples/fixed_samples_$w_dim-dim.csv"
        end
    end

    # filename = "high_prec_samples.csv"

    writedlm(filename, results, ',')
end

function time_measure(its, w_dim, h, err)
    W = [1.0,-0.5,-1.2,-0.3,0.7,0.2,-0.9,0.1,1.7]
    W = W[1:w_dim]
    s = 0
    for i in 1:its
        lev = iterated_integrals(W,h,err)
    end

    println(s)
end

function generate_all()
    for i in 2:8
        gen_samples(its = 1048576, w_dim = i)
        gen_samples(w_dim = i, fixed = true)
        println("$i done")
    end
end

function generate_unfixed_test_samples()
    for i in 2:8
        gen_samples(its = 65536, w_dim = i, filename = "samples/non-fixed_test_samples_$i-dim.csv")
        println("$i done")
    end
end

function gen_all_fixed_2d()

    W:: Array{Float64} = [1.0,-0.5,-1.2,-0.3,0.7,0.2,-0.9,0.1,1.7]
    idx = 1

    for k in 1:9
        for l in (k+1):9
            results = Array{Float64}(undef,65536,3)

            w = [W[k],W[l]]
            for i in 1:65536
                II = iterated_integrals(w, 1.0, 0.0001)
                a = 0.5* (II[1,2] - II[2,1])
                results[i,:] = [W[k],W[l],a]
            end

            filename = "samples/fixed_samples_2-dim$idx.csv"
            writedlm(filename, results, ',')
            println("($(W[k]), $(W[l])) done")

            idx += 1
        end
    end

end

function gen_all_fixed_3d(;max_idx::Int64 = 10)

    W:: Array{Float64} = [1.0,-0.5,-1.2,-0.3,0.7,0.2,-0.9,0.1,1.7]
    idx = 1
    for k in 1:3:9
        for l in (k+1):2:9
            for m in (l+1):(l+1)
                if idx > max_idx
                    break
                end
                if m > 9
                    break
                end

                w = [W[k], W[l], W[m]]
                filename = "samples/fixed_samples_3-dim$idx.csv"
                gen_samples(its = 1048576, w_dim = 3, fixed = true, W = w, filename = filename)

                println("($(W[k]), $(W[l]), $(W[m])) done")

                idx += 1
            end
        end
    end

end

function list_all_3d_combos()
    w_dim = 3
    resDim = Int64(9*(9-1)*(9-2)/6)
    lst:: Array{Array{Float64}} = Array{Array{Float64}}(undef,resDim)
    W:: Array{Float64} = [1.0,-0.5,-1.2,-0.3,0.7,0.2,-0.9,0.1,1.7]
    idx = 1
    for k in 1:9
        for l in (k+1):9
            for m in (l+1):9

                # results = Array{Float64}(undef,65536,3)

                # w = [W[k],W[l]]
                # for i in 1:65536
                #     II = iterated_integrals(w, 1.0, 0.0001)
                #     a = 0.5* (II[1,2] - II[2,1])
                #     results[i,:] = [W[k],W[l],a]
                # end

                # filename = "samples/fixed_samples_3-dim$idx.csv"
                # writedlm(filename, results, ',')
                lst[idx] = [W[k], W[l], W[m]]

                idx += 1
            end
        end
    end
    println(lst)
end


function list_pairs(w_dim)
    resDim = Int64(w_dim*(w_dim-1)/2)
    res = Array{Tuple{Int64,Int64}}(undef,resDim)
    idx:: Int = 1
    for k in 1:w_dim
        for l in (k+1):w_dim
            res[idx] = (k,l)
            idx +=1
        end
    end
    println(res)
end


function gen_all_fixed_4d(;max_idx::Int64 = 30)

    W:: Array{Float64} = [1.0,-0.5,-1.2,-0.3,0.7,0.2,-0.9,0.1,1.7,1.1,-2.3,0.05,-1.4]
    idx = 1
    for k in 1:5:13
        for l in (k+1):3:13
            for m in (l+1):(l+1)

                if m > 13
                    break
                end
                for n in (m+1):(m+1)
                    if idx > max_idx
                        break
                    end
                    if n > 13
                        break
                    end

                    w = [W[k], W[l], W[m], W[n]]
                    filename = "samples/fixed_samples_4-dim$(idx+6).csv"
                    gen_samples(its = 1048576, w_dim = 4, err = 0.0001, fixed = true, W = w, filename = filename)

                    println("($(W[k]), $(W[l]), $(W[m]), $(W[n])) done")

                    idx += 1
                end
            end
        end
    end

end

function gen_all_fixed_5d(;max_idx::Int64 = 30)

    W:: Array{Float64} = [1.0,-0.5,-1.2,-0.3,0.7,0.2,-0.9,0.1,1.7,3.0,-2.3,0.05,-4.2,-1.5]
    idx = 1
    for k in 1:5:13
        for l in (k+1):3:13
            for m in (l+1):(l+1)

                if m > 13
                    break
                end
                for n in (m+1):(m+1)
                    if idx > max_idx
                        break
                    end
                    if n > 13
                        break
                    end
                    w = [W[k], W[l], W[m], W[n], W[n+1]]
                    filename = "samples/fixed_samples_5-dim$idx.csv"
                    gen_samples(its = 1048576, w_dim = 5, err = 0.0001, fixed = true, W = w, filename = filename)

                    println("($(W[k]), $(W[l]), $(W[m]), $(W[n]), $(W[n+1])) done")

                    idx += 1
                end
            end
        end
    end

end

function cora_explain_dataset(modelfile, datasetfile, epsilon, resultsDir, varargin)
% cora_explain_dataset - explains the samples of a dataset of SpExplAIn
%    using neuralNetwork/explain, one sample at a time, with the results
%    in the same format as the VeriX-plus scripts of SpExplAIn
%    (data/scripts/tacas27/verix/verix1.py, read by to_phi.py)
%
% Syntax:
%    cora_explain_dataset(modelfile, datasetfile, epsilon, resultsDir, varargin)
%
% Inputs:
%    modelfile - char, network in the .nnet format
%    datasetfile - char, CSV with a header, the features (scaled to [0,1],
%       channel after channel for images) and the label in the last column
%    epsilon - numeric, perturbation radius
%    resultsDir - char, directory of the results
%    varargin - name-value pairs
%       <'InputSize',inputSize> - numeric, [H,W,C] (default: [n,1,1])
%       <'Timeout',timeout> - numeric, timeout per explanation in seconds
%       <'Method',method> - char, see neuralNetwork/explain
%       <'FeatOrder',featOrder> - char, see neuralNetwork/explain
%       <'InputBounds',inputBounds> - numeric, see neuralNetwork/explain
%       <'Samples',samples> - numeric, indices of the samples (from 0)
%       <'Resume',resume> - logical, skip the samples with results
%       <'Verbose',verbose> - logical, verbose output of explain
%       <'WarmUp',warmUp> - logical, first explain the first sample to run
%          once without results (default: true), so that the JIT compiler
%          of MATLAB, which runs in extra threads, compiles the code before
%          the measured explanations
%    (as an application compiled by build_cora_explain, all arguments are
%    text, e.g., InputSize "[28 28 1]" or Timeout 600)
%
% Outputs:
%    - (results of the sample <idx> (from 0) in resultsDir/<idx>/:
%       sat.txt and unsat.txt - the fixed and the freed pixels (from 0),
%       robust.txt - all pixels can be freed,
%       time.txt - the runtime in seconds, missing if timed out,
%       timeout.txt or error.txt - no explanation,
%       explanation.png - the freed pixels in red (images only);
%       and one row per sample in resultsDir/summary.csv)
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: neuralNetwork/explain

% Authors:       Tomas Kolarik
% Written:       03-November-2026
% Last update:   04-November-2026 (warm-up)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% parse inputs
data = readmatrix(datasetfile,'NumHeaderLines',1);
X = data(:,1:end-1)';
labels = data(:,end)';
[nFeat,nSamples] = size(X);

p = inputParser;
p.addParameter('InputSize',[nFeat,1,1]);
p.addParameter('Timeout',Inf);
p.addParameter('Method','abstract+refine');
p.addParameter('FeatOrder','sensitivity');
p.addParameter('InputBounds',[0 1]);
p.addParameter('Samples',0:nSamples-1);
p.addParameter('Resume',false);
p.addParameter('Verbose',false);
p.addParameter('WarmUp',true);
p.parse(varargin{:});
opts = p.Results;

% the arguments of a compiled application are text
if ischar(epsilon) || isstring(epsilon)
    epsilon = str2double(epsilon);
end
for name = ["InputSize","Timeout","InputBounds","Samples","Resume","Verbose","WarmUp"]
    if ischar(opts.(name)) || isstring(opts.(name))
        opts.(name) = str2num(char(opts.(name))); %#ok<ST2NM>
    end
end
opts.Resume = logical(opts.Resume);
opts.Verbose = logical(opts.Verbose);
opts.WarmUp = logical(opts.WarmUp);
inputSize = opts.InputSize;
nPixels = inputSize(1)*inputSize(2);
if nPixels*inputSize(3) ~= nFeat
    throw(CORAerror('CORA:wrongValue','name-value pair ''InputSize''', ...
        sprintf('has to have %i features in total',nFeat)))
end

% load network
nn = neuralNetwork.readNNetNetwork(modelfile);
% readNNetNetwork applies the activation also after the output layer,
% but the outputs of the networks are the logits
nn = neuralNetwork(nn.layers(1:end-1));
nn = nn.getNormalForm();
% a binary classifier with one output y (class 1 iff y >= 0, as in SpExplAIn)
% would be a regression task for explain: use the logits [0; y] instead
isBinary = nn.neurons_out == 1;
if isBinary
    nn.layers{end+1} = nnLinearLayer([0; 1],[0; 0]);
    nn = nn.getNormalForm();
end

if ~isfolder(resultsDir)
    mkdir(resultsDir);
end
summaryFile = fullfile(resultsDir,'summary.csv');
if ~isfile(summaryFile)
    aux_writeLines(summaryFile,"idx,label,predicted,fixed,pixels,time,status");
end

fprintf("%s, %s, epsilon=%g: %i samples\n",modelfile,datasetfile,epsilon,numel(opts.Samples));

% the samples to run
isDone = arrayfun(@(idx) opts.Resume && aux_isDone(fullfile(resultsDir,num2str(idx))),opts.Samples);
samples = opts.Samples(~isDone);

if opts.WarmUp && ~isempty(samples)
    % the same explanation as of the first sample, without results
    x = X(:,samples(1)+1);
    timerVal = tic;
    try
        nn.reset();
        aux_explain(nn,x,aux_predict(nn,x,isBinary),epsilon,inputSize,opts);
    catch ME
        % reported by the measured explanation of the sample
        fprintf("warm-up failed: %s\n",ME.message);
    end
    fprintf("warm-up: %.3f s\n",toc(timerVal));
end

for idx = samples
    sampleDir = fullfile(resultsDir,num2str(idx));
    if isfolder(sampleDir)
        rmdir(sampleDir,'s');
    end
    mkdir(sampleDir);

    x = X(:,idx+1);
    predicted = aux_predict(nn,x,isBinary);

    nFixed = NaN;
    t = NaN;
    try
        nn.reset();
        timerVal = tic;
        [idxFreed,timesPerFeat] = aux_explain(nn,x,predicted,epsilon,inputSize,opts);
        t = toc(timerVal);

        % explain stops early if it would exceed the timeout
        if any(isnan(timesPerFeat)) || t >= opts.Timeout
            status = "timeout";
            aux_writeLines(fullfile(sampleDir,'timeout.txt'),t);
        else
            idxFixed = setdiff(1:nPixels,idxFreed);
            nFixed = numel(idxFixed);
            if isempty(idxFixed)
                status = "robust";
                aux_writeLines(fullfile(sampleDir,'robust.txt'),idx);
            else
                status = "explained";
                aux_writeLines(fullfile(sampleDir,'sat.txt'),idxFixed-1);
                aux_writeLines(fullfile(sampleDir,'unsat.txt'),sort(idxFreed)-1);
            end
            aux_writeLines(fullfile(sampleDir,'time.txt'),sprintf('%.6f',t));
            aux_writeImage(fullfile(sampleDir,'explanation.png'),x,idxFreed,inputSize);
        end
    catch ME
        status = "error";
        aux_writeLines(fullfile(sampleDir,'error.txt'),getReport(ME,'extended','hyperlinks','off'));
    end

    fprintf("%i: label %i, predicted %i, fixed %g/%i, time %.3f s, %s\n", ...
        idx,labels(idx+1),predicted-1,nFixed,nPixels,t,status);
    aux_appendLines(summaryFile,sprintf("%i,%i,%i,%g,%i,%.6f,%s", ...
        idx,labels(idx+1),predicted-1,nFixed,nPixels,t,status));
end

end


% Auxiliary functions -----------------------------------------------------

function predicted = aux_predict(nn,x,isBinary)
    % the prediction of the network, which is explained (the classes are from 1)
    y = nn.evaluate(x);
    if isBinary
        % the tie y = 0 is class 1
        predicted = 1 + (y(2) >= y(1));
    else
        [~,predicted] = max(y);
    end
end

function [idxFreed,timesPerFeat] = aux_explain(nn,x,predicted,epsilon,inputSize,opts)
    [idxFreed,~,timesPerFeat] = nn.explain(x,predicted,epsilon, ...
        'Method',opts.Method,'FeatOrder',opts.FeatOrder,'InputSize',inputSize, ...
        'Timeout',opts.Timeout,'InputBounds',opts.InputBounds,'Verbose',opts.Verbose,'Plot',false);
end

function res = aux_isDone(sampleDir)
    % the sample has a result (also a timeout)
    res = isfile(fullfile(sampleDir,'time.txt')) || isfile(fullfile(sampleDir,'timeout.txt'));
end

function aux_writeLines(file,lines)
    % one line per element (a char array is one line)
    lines = string(lines);
    writelines(lines(:),file);
end

function aux_appendLines(file,lines)
    lines = string(lines);
    writelines(lines(:),file,'WriteMode','append');
end

function aux_writeImage(file,x,idxFreed,inputSize)
    % image of the sample with the freed pixels in red
    H = inputSize(1); W = inputSize(2); C = inputSize(3);
    if H == 1 || W == 1
        % tabular data
        return
    end

    % the features of each channel are row after row
    img = reshape(x,[W,H,C]);
    img = permute(img,[2 1 3]);
    if C == 1
        img = repmat(img,[1 1 3]);
    end
    isFreed = false(W,H);
    isFreed(idxFreed) = true; % column-major index of the transposed image
    isFreed = isFreed';
    img = reshape(img,[H*W,3]);
    img(isFreed(:),:) = repmat([1 0 0],nnz(isFreed),1);
    img = reshape(img,[H,W,3]);

    % upscale without the image processing toolbox
    scale = 8;
    img = repelem(img,scale,scale);
    imwrite(img,file);
end

% ------------------------------ END OF CODE ------------------------------

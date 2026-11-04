function build_cora_explain(coraDir, outDir)
% build_cora_explain - compiles cora_explain_dataset into a standalone
%    application that runs on the MATLAB Runtime (no MATLAB license needed)
%
% Syntax:
%    build_cora_explain(coraDir, outDir)
%
% Inputs:
%    coraDir - char, CORA with the patch cora-explain.patch
%              (see build.sh, which also runs this)
%    outDir - char, output directory
%
% Outputs:
%    - (outDir/cora_explain_dataset and outDir/run_cora_explain_dataset.sh,
%       run as: run_cora_explain_dataset.sh <runtime root> <arguments>,
%       see cora_explain_dataset)
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: cora_explain_dataset

% Authors:       Tomas Kolarik
% Written:       03-November-2026
% Last update:   04-November-2026 (without the Parallel Computing Toolbox;
%                   CORA given, as this is outside of it)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

coraDir = char(coraDir);
if ~isfolder(coraDir)
    error("CORA not found: %s", coraDir);
end
if ~isfolder(outDir)
    mkdir(outDir);
end

% the dependency analysis misses the functions reached only by name
% (e.g., layer types and set operations), so include the whole folders
folders = {'nn','contSet','global','matrixSet','specification'};
addArgs = {};
for i = 1:numel(folders)
    addArgs = [addArgs, {'-a', fullfile(coraDir,folders{i})}]; %#ok<AGROW>
end

oldPath = path;
restorePath = onCleanup(@() path(oldPath));
% without the Parallel Computing Toolbox: CORA only checks for a GPU
% (gpuDeviceCount), but the toolbox would require its GPU runtime addon,
% which the MATLAB Runtime images do not have
parallelDirs = {fullfile(matlabroot,'toolbox','parallel'), ...
    fullfile(matlabroot,'toolbox','shared','parallel')};
pathDirs = strsplit(path,pathsep);
isExcluded = startsWith(pathDirs,parallelDirs);
if any(isExcluded)
    rmpath(pathDirs{isExcluded});
end

% one thread and no Java, as the experiments run many processes in parallel
mcc('-m', fullfile(fileparts(mfilename('fullpath')),'cora_explain_dataset.m'), ...
    '-R', '-nojvm', '-R', '-singleCompThread', ...
    addArgs{:}, '-d', outDir, '-v');

fprintf("Built: %s\n", outDir);

end

% ------------------------------ END OF CODE ------------------------------

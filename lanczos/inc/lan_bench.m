%
%  Copyright 2013 William J. Brouwer, Pierre-Yves Taunay
%
%  Licensed under the Apache License, Version 2.0 (the "License");
%  you may not use this file except in compliance with the License.
%  You may obtain a copy of the License at
%
%      http://www.apache.org/licenses/LICENSE-2.0
%
%  Unless required by applicable law or agreed to in writing, software
%  distributed under the License is distributed on an "AS IS" BASIS,
%  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%  See the License for the specific language governing permissions and
%  limitations under the License.
%




function [out] = lan_bench(sz,procs)

%develops a test set for working with lanczos solver
%random, complex hermitian matrix


a=tril(rand(sz,sz)+i*rand(sz,sz));

out = a+a';

aa=real(out);
bb=imag(out);

intrlv = reshape([aa(:) bb(:)]',2*size(aa,1), []);

chunk = sz*2/procs;

for i=0:procs-1

bin(:,i*sz+1:(i+1)*sz) = intrlv(i*chunk+1:(i+1)*chunk,:);


end



label = ['test_lan_',num2str(sz),'X',num2str(sz),'_',num2str(procs),'procs'];

matf = [label,'.mat'];
save(matf,"out");


binf = [label,'.bin'];
fid = fopen(binf,'w');
fwrite(fid,bin,'double');

fclose(fid);






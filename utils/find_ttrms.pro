pro find_ttrms, directory, outfile=outfile

;;WARNING: The following should relate to the AO frequency
SMOOTH_IT = 201
if not keyword_set(outfile) then outfile='ttrms.csv'

spawn, 'ls '+ directory + '/n*sav', files

if (strlen(files[0]) eq 0) then begin
  print, "WARNING: no files found"
  return
endif

openw, 1, outfile

for i=0,n_elements(files)-1 do begin
  filestr = strmid(files[i], strpos(files[i], '/', /reverse_search) + 1)
  filestr = strmid(filestr, 0, strpos(filestr, '_'))
  restore, files[i]
  if n_elements(a.timestamp) gt 500 then begin
   tt_fast = a.ttcommands - smooth(a.ttcommands, [1, SMOOTH_IT], /edge_trunc)
   tt_fast_rms = sqrt(mean(tt_fast^2))
  endif else tt_fast_rms=-1
  printf, 1, filestr, tt_fast_rms, format='(A6,", ",F6.3)'
endfor

close, 1

end

#!/usr/bin/perl

$input_file_name = "system_estimate_T";
$output_file_name = "area_info";
open (FILE_OUTPUT, ">$output_file_name.csv");

for ($n = 0; $n <= 14; $n++) {
#for ($n = 2; $n <= 2; $n++) {

    $FF = 0; $LUT = 0; $DSP = 0; $BRAM = 0;
    $T = 2 ** $n;
    if ( open (FILE_INPUT, "./../${input_file_name}${T}.xtxt") ) {
        $line = <FILE_INPUT>;
        while ($line ne '') {
            $symbol = $line =~ /FF/;
            if ($symbol) {
                #print("$line\n");
                $line = <FILE_INPUT>;
                $flag = 1;
                #print("flag = $flag\n");
                while ($flag eq '1') {
                    $line = <FILE_INPUT>;
                    $flag = $line =~ /\|\ (\S+)\ *\|\ (\S+)\ *\|\ (\S+)\ *\|\ (\S+)\ *\|\ (\S+)\ *\|\ (\S+)\ *\|/;
                    if ($flag) {
                        #print("symbol = $symbol\n");
                        #print("$1\n$2\n$3\n$4\n$5\n$6\n");
                        $FF += $3;  # FF
                        $LUT += $4;  # LUT
                        $DSP += $5;  # DSP
                        $BRAM += $6;  # BRAM
                    }
                }
                last;
            }
            $line = <FILE_INPUT>;
        }
        print("T$T: $FF, $LUT, $DSP, $BRAM\n");
        print FILE_OUTPUT ("T$T, $FF, $LUT, $DSP, $BRAM\n");
    }
}

:- dynamic parameter_pos/1.
:- dynamic parameter_neg/1.

% Kesehatan
kesehatan("Kesehatan Plus").
kesehatan("Kesehatan Normal").
kesehatan("Kesehatan Negatif").

% Aturan Kesehatan Negatif
aturan_kesehatan("Kesehatan Negatif") :-
    parameter_pos(bmi_overweight);
    parameter_pos(bmi_underweight),
    parameter_pos(aktivitas_sangat_rendah);
    parameter_pos(intake_berlebih);
    parameter_pos(intake_kurang);
    parameter_pos(stres_tinggi);
    parameter_pos(merokok);
    parameter_pos(tidur_kurang).

% Aturan Kesehatan Normal
aturan_kesehatan("Kesehatan Normal") :-
    \+ parameter_pos(bmi_overweight),
    \+ parameter_pos(bmi_underweight),
    parameter_pos(aktivitas_moderat),
    parameter_pos(intake_seimbang),
    \+ parameter_pos(stres_tinggi),
    \+ parameter_pos(merokok),
    parameter_pos(tidur_cukup).

% Aturan Kesehatan Plus
aturan_kesehatan("Kesehatan Plus") :-
    \+ parameter_pos(bmi_overweight),
    \+ parameter_pos(bmi_underweight),
    parameter_pos(aktivitas_tinggi),
    parameter_pos(intake_seimbang),
    \+ parameter_pos(stres_tinggi),
    \+ parameter_pos(merokok),
    parameter_pos(tidur_cukup).

% Fakta tambahan terkait parameter kesehatan
parameter_pos(stres_tinggi).
parameter_pos(merokok).
parameter_pos(tidur_kurang).
parameter_pos(tidur_cukup).
parameter_pos(aktivitas_tinggi).
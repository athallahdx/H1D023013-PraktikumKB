from tkinter import *
from pyswip import Prolog

def hitung_dan_diagnosa():
    try:
        berat = float(entry_berat.get())
        tinggi = float(entry_tinggi.get()) 
        umur = int(entry_umur.get())
        gender_val = gender_var.get()
        aktivitas_val = aktivitas_var.get()
        intake_val = float(entry_intake.get())
        stres_val = stres_var.get()
        merokok_val = merokok_var.get()
        tidur_val = tidur_var.get()

        tinggi_meter = tinggi / 100 
        bmi = berat / (tinggi_meter * 2)

        if gender_val == "Male":
            tdee = (10 * berat) + (6.25 * tinggi) - (5 * umur) + 5 + 400
        else:  
            tdee = (10 * berat) + (6.25 * tinggi) - (5 * umur) - 16 + 400

        if intake_val < tdee - 500:
            intake_category = "Kurang"
        elif intake_val > tdee + 500:
            intake_category = "Berlebih"
        else:
            intake_category = "Seimbang"

        prolog = Prolog()
        prolog.consult("pakar_kesehatan.pl")

        list(prolog.query("retractall(parameter_pos(_))"))
        list(prolog.query("retractall(parameter_neg(_))"))

        if bmi < 18.5:
            prolog.assertz("parameter_pos(bmi_underweight)")
        elif bmi > 25:
            prolog.assertz("parameter_pos(bmi_overweight)")

        if aktivitas_val == "Rendah":
            prolog.assertz("parameter_pos(aktivitas_sangat_rendah)")
        elif aktivitas_val == "Moderat":
            prolog.assertz("parameter_pos(aktivitas_moderat)")
        elif aktivitas_val == "Tinggi":
            prolog.assertz("parameter_pos(aktivitas_tinggi)")

        if intake_category == "Kurang":
            prolog.assertz("parameter_pos(intake_kurang)")
        elif intake_category == "Seimbang":
            prolog.assertz("parameter_pos(intake_seimbang)")
        elif intake_category == "Berlebih":
            prolog.assertz("parameter_pos(intake_berlebih)")

        if stres_val == "Tinggi":
            prolog.assertz("parameter_pos(stres_tinggi)")

        if merokok_val == "Ya":
            prolog.assertz("parameter_pos(merokok)")

        if tidur_val == "Kurang":
            prolog.assertz("parameter_pos(tidur_kurang)")
        elif tidur_val == "Cukup":
            prolog.assertz("parameter_pos(tidur_cukup)")

        hasil = list(prolog.query("aturan_kesehatan(X)"))
        if hasil:
            label_hasil.config(text=f"Hasil Diagnosa: {hasil[0]['X']}")
        else:
            label_hasil.config(text="Hasil Diagnosa: Tidak Diketahui")

    except Exception as e:
        label_hasil.config(text=f"Error: {str(e)}")

root = Tk()
root.title("Sistem Pakar Kesehatan")
root.geometry("500x700")

font_label = ("Arial", 12)
font_entry = ("Arial", 12)
font_button = ("Arial", 12, "bold")
font_title = ("Arial", 14, "bold")

Label(root, text="Diagnosa Kesehatan Berdasarkan BMI", font=font_title, pady=10).grid(row=0, column=0, columnspan=2)

Label(root, text="Berat Badan (kg):", font=font_label).grid(row=1, column=0, sticky=W, padx=10, pady=5)
entry_berat = Entry(root, font=font_entry)
entry_berat.grid(row=1, column=1, padx=10, pady=5)

Label(root, text="Tinggi Badan (cm):", font=font_label).grid(row=2, column=0, sticky=W, padx=10, pady=5)
entry_tinggi = Entry(root, font=font_entry)
entry_tinggi.grid(row=2, column=1, padx=10, pady=5)

Label(root, text="Umur (tahun):", font=font_label).grid(row=3, column=0, sticky=W, padx=10, pady=5)
entry_umur = Entry(root, font=font_entry)
entry_umur.grid(row=3, column=1, padx=10, pady=5)

Label(root, text="Jenis Kelamin:", font=font_label).grid(row=4, column=0, sticky=W, padx=10, pady=5)
gender_var = StringVar(root)
gender_var.set("Male")
OptionMenu(root, gender_var, "Male", "Female").grid(row=4, column=1, padx=10, pady=5)

Label(root, text="Aktivitas Harian:", font=font_label).grid(row=5, column=0, sticky=W, padx=10, pady=5)
aktivitas_var = StringVar(root)
aktivitas_var.set("Moderat")
OptionMenu(root, aktivitas_var, "Rendah", "Moderat", "Tinggi").grid(row=5, column=1, padx=10, pady=5)

Label(root, text="Intake Kalori Harian:", font=font_label).grid(row=6, column=0, sticky=W, padx=10, pady=5)
entry_intake = Entry(root, font=font_entry)
entry_intake.grid(row=6, column=1, padx=10, pady=5)

Label(root, text="Tingkat Stres:", font=font_label).grid(row=7, column=0, sticky=W, padx=10, pady=5)
stres_var = StringVar(root)
stres_var.set("Rendah")
OptionMenu(root, stres_var, "Rendah", "Tinggi").grid(row=7, column=1, padx=10, pady=5)

Label(root, text="Apakah Anda Merokok?", font=font_label).grid(row=8, column=0, sticky=W, padx=10, pady=5)
merokok_var = StringVar(root)
merokok_var.set("Tidak")
OptionMenu(root, merokok_var, "Ya", "Tidak").grid(row=8, column=1, padx=10, pady=5)

Label(root, text="Jumlah Tidur (Jam):", font=font_label).grid(row=9, column=0, sticky=W, padx=10, pady=5)
tidur_var = StringVar(root)
tidur_var.set("Cukup")
OptionMenu(root, tidur_var, "Kurang", "Cukup").grid(row=9, column=1, padx=10, pady=5)

Button(root, text="Diagnosa", command=hitung_dan_diagnosa, font=font_button, bg="lightblue").grid(row=10, columnspan=2, pady=20)

label_hasil = Label(root, text="Hasil Diagnosa: -", font=("Arial", 12, "bold"), fg="green")
label_hasil.grid(row=11, columnspan=2, pady=10)

root.mainloop()

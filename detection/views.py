from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import render
from django.shortcuts import redirect
from django.urls import reverse
from  django.core.files.storage import FileSystemStorage
import datetime

from .models import *
import os

def first(request):
    return render(request,'index.html')

def index(request):
    return render(request,'index.html')

def register(request):
    return render(request,'register.html')

def registration(request):
    if request.method=="POST":
        name=request.POST.get('name')
        email=request.POST.get('email')
        password=request.POST.get('password')
        if registerr.objects.filter(email=email).exists():
            return render(request,'register.html',{'message':'Email already exist'})
        reg=registerr(name=name,email=email,password=password)
        reg.save()
    return render(request,'index.html' ,{'message':'Register Success'})

def login(request):
    return render(request,'login.html')

def addlogin(request):
    email = request.POST.get('email')
    password = request.POST.get('password')
    if email == 'admin@gmail.com' and password =='admin':
        request.session['logintdetail'] = email
        request.session['admin'] = 'admin'
        return render(request,'index.html')

    elif registerr.objects.filter(email=email,password=password).exists():
        userdetails=registerr.objects.get(email=request.POST['email'], password=password)
        if userdetails.password == request.POST['password']:
            request.session['uid'] = userdetails.id
            request.session['uname'] = userdetails.name
        
        return render(request,'index.html')
        
    else:
        return render(request, 'login.html', {'success':'Invalid email id or Password'})
    
def logout(request):
    session_keys = list(request.session.keys())
    for key in session_keys:
        del request.session[key]
    return redirect(first)

def v_users(request):
    user = registerr.objects.all()
    return render(request, 'viewusers.html', {'result': user})

def test(request):
    return render(request,'test.html')

def addfile(request):
    if request.method=="POST":
        u_id = request.session['uid']
        os.system("python ML/detection.py")

    return render(request,'test.html')
    
def v_result(request):
    uid=request.session['uid']
    user = uploads.objects.filter(u_id=uid)
    return render(request, 'viewresult.html', {'result': user})

def feedback(request):
    if request.method=="POST":
        feedback=request.POST.get('feedback')
        user=request.session['uname']
    
        reg=feedbacks(feedback=feedback,user=user)
        reg.save()
        return render(request,'index.html',{'message':'Thank you for your feedback'})
    else:
        return render(request,'feedback.html')
    
def v_feedback(request):
    user = feedbacks.objects.all()
    return render(request, 'v_feedback.html', {'result': user})

def profile(request):
    u=request.session['uid']
    user = registerr.objects.get(id=u)
    return render(request, 'profile.html', {'result': user})

def editpro(request,id):
    user = registerr.objects.get(id=id)
    return render(request, 'editpro.html', {'result': user})

def editproadd(request,id):
    if request.method=="POST":
        name=request.POST.get('name')
        email=request.POST.get('email')
        password=request.POST.get('password')
        reg=registerr(name=name,email=email,password=password,id=id)
        reg.save()
    return redirect(profile)